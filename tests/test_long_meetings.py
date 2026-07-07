"""Tests for chaining scripted meeting sections into long recordings."""

from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.data.dialogue_scripts import DialogueScript, Turn
from distil_vibevoice.data.long_meetings import (
    MIN_TURN_S,
    SECONDS_PER_CHAR,
    build_long_meeting,
)
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder

SR = 16000


def _script(speakers: list[str], lines: list[tuple[str, str]], domain: str) -> DialogueScript:
    return DialogueScript(
        speakers=speakers,
        turns=[Turn(spk, text) for spk, text in lines],
        domain=domain,
    )


def _two_sections() -> tuple[list[DialogueScript], dict[str, np.ndarray]]:
    a_lines = [("Alice", "hello everyone lets begin the meeting today"),
               ("Bob", "sure I have a quick status update to share"),
               ("Alice", "great please go ahead with the numbers")]
    b_lines = [("Bob", "continuing from before the deadline is firm"),
               ("Alice", "understood we will align on the timeline"),
               ("Bob", "thanks that closes my items for now")]
    scripts = [_script(["Alice", "Bob"], a_lines, "engineering_sync"),
               _script(["Alice", "Bob"], b_lines, "engineering_sync")]
    speaker_wavs = {"Alice": np.zeros(SR, np.float32), "Bob": np.zeros(SR, np.float32)}
    return scripts, speaker_wavs


def _sum_turn_durations(scripts: list[DialogueScript]) -> float:
    total = 0.0
    for s in scripts:
        for turn in s.turns:
            total += max(len(turn.text) * SECONDS_PER_CHAR, MIN_TURN_S)
    return total


def test_basic_shapes_and_bounds() -> None:
    scripts, wavs = _two_sections()
    wav, segs = build_long_meeting(scripts, wavs, SR, rng=np.random.default_rng(0))
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert np.all(np.isfinite(wav))
    assert len(segs) == sum(len(s.turns) for s in scripts)
    total_s = len(wav) / SR
    for prev, cur in zip(segs, segs[1:]):
        assert cur.start >= prev.end - 1e-9
    for s in segs:
        assert 0.0 <= s.start < s.end <= total_s + 1e-9


def test_total_duration_is_sections_plus_breaks() -> None:
    scripts, wavs = _two_sections()
    lo, hi = 30.0, 120.0
    wav, segs = build_long_meeting(
        scripts, wavs, SR, break_range=(lo, hi), rng=np.random.default_rng(1)
    )
    total_s = len(wav) / SR
    turn_s = _sum_turn_durations(scripts)
    n_breaks = len(scripts) - 1
    gap = total_s - turn_s
    # The only thing between sections is the silence break(s).
    assert n_breaks * lo - 0.05 <= gap <= n_breaks * hi + 0.05


def test_segments_within_audio_bounds_many_sections() -> None:
    scripts, wavs = _two_sections()
    scripts = scripts * 3  # six sections
    wav, segs = build_long_meeting(scripts, wavs, SR, rng=np.random.default_rng(2))
    total_samples = len(wav)
    for s in segs:
        i0 = int(round(s.start * SR))
        i1 = int(round(s.end * SR))
        assert 0 <= i0 < i1 <= total_samples


def test_deterministic_with_seeded_rng() -> None:
    scripts, wavs = _two_sections()
    wav_a, segs_a = build_long_meeting(scripts, wavs, SR, rng=np.random.default_rng(7))
    wav_b, segs_b = build_long_meeting(scripts, wavs, SR, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(wav_a, wav_b)
    assert segs_a == segs_b


def test_shared_speaker_keeps_signature_across_sections() -> None:
    # Two sections, speakers Alice & Bob recur.  Same-speaker cross-section
    # cosine must beat cross-speaker cosine (the embedder can re-identify).
    scripts, wavs = _two_sections()
    wav, segs = build_long_meeting(scripts, wavs, SR, rng=np.random.default_rng(3))
    emb = MfccStatsEmbedder()

    def section_bounds() -> list[tuple[float, float]]:
        # Section i spans from its first to last segment; a gap separates them.
        bounds = []
        sec_start = segs[0].start
        prev_end = segs[0].end
        for s in segs[1:]:
            if s.start > prev_end + 1e-6:  # crossed a silence break
                bounds.append((sec_start, prev_end))
                sec_start = s.start
            prev_end = s.end
        bounds.append((sec_start, prev_end))
        return bounds

    bounds = section_bounds()
    assert len(bounds) == 2

    def embed(speaker: str, sec: int) -> np.ndarray:
        lo, hi = bounds[sec]
        pieces = [
            wav[int(round(s.start * SR)):int(round(s.end * SR))]
            for s in segs
            if s.speaker == speaker and lo - 1e-6 <= s.start and s.end <= hi + 1e-6
        ]
        return emb.embed(np.concatenate(pieces), SR)

    alice0, alice1 = embed("Alice", 0), embed("Alice", 1)
    bob0, bob1 = embed("Bob", 0), embed("Bob", 1)

    same_alice = float(alice0 @ alice1)
    same_bob = float(bob0 @ bob1)
    cross = max(float(alice0 @ bob1), float(bob0 @ alice1))

    assert same_alice > cross
    assert same_bob > cross


def test_empty_scripts() -> None:
    wav, segs = build_long_meeting([], {}, SR)
    assert wav.size == 0
    assert segs == []


def test_tts_fn_is_used_when_given() -> None:
    scripts, wavs = _two_sections()
    marker = {}

    def tts_fn(text: str, ref_wav: np.ndarray) -> np.ndarray:
        marker["called"] = True
        return np.full(int(0.5 * SR), 0.1, dtype=np.float32)

    wav, segs = build_long_meeting(scripts, wavs, SR, tts_fn=tts_fn,
                                   rng=np.random.default_rng(4))
    assert marker.get("called")
    for s in segs:
        assert s.end - s.start == pytest.approx(0.5, abs=1e-3)


def test_rejects_bad_break_range() -> None:
    scripts, wavs = _two_sections()
    with pytest.raises(ValueError):
        build_long_meeting(scripts, wavs, SR, break_range=(120.0, 30.0))
