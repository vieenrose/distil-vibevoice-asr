"""Tests for distil_vibevoice.data.simulate_meetings (CPU-only, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.data.simulate_meetings import simulate_meeting

SR = 16000


def _sine(freq: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _utterances(n: int = 8) -> list[tuple[np.ndarray, str, str]]:
    return [
        (_sine(200.0 + 50.0 * i, 0.5 + 0.1 * (i % 3)), f"spk{i % 3}", f"utt {i}")
        for i in range(n)
    ]


def test_basic_output() -> None:
    utts = _utterances()
    wav, segs = simulate_meeting(utts, SR, overlap_ratio=0.1, rng=np.random.default_rng(0))
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert len(segs) == len(utts)
    assert np.all(np.isfinite(wav))
    assert np.max(np.abs(wav)) <= 1.0


def test_segments_sorted_and_durations_match() -> None:
    utts = _utterances(10)
    _, segs = simulate_meeting(utts, SR, overlap_ratio=0.3, rng=np.random.default_rng(1))
    starts = [s.start for s in segs]
    assert starts == sorted(starts)
    ref_durs = sorted(len(w) / SR for w, _, _ in utts)
    got_durs = sorted(s.end - s.start for s in segs)
    np.testing.assert_allclose(got_durs, ref_durs, atol=1e-6)


def test_speakers_and_text_preserved() -> None:
    utts = _utterances(6)
    _, segs = simulate_meeting(utts, SR, overlap_ratio=0.0, rng=np.random.default_rng(2))
    # With zero overlap, order is sequential, so fields line up one-to-one.
    assert [s.speaker for s in segs] == [spk for _, spk, _ in utts]
    assert [s.text for s in segs] == [txt for _, _, txt in utts]


def test_zero_overlap_ratio_means_no_overlap() -> None:
    utts = _utterances(12)
    _, segs = simulate_meeting(utts, SR, overlap_ratio=0.0,
                               silence_range=(0.1, 0.4), rng=np.random.default_rng(3))
    for prev, cur in zip(segs, segs[1:]):
        assert cur.start >= prev.end - 1e-9


def test_half_overlap_ratio_produces_some_overlap() -> None:
    utts = _utterances(20)
    _, segs = simulate_meeting(utts, SR, overlap_ratio=0.5, rng=np.random.default_rng(4))
    overlaps = sum(1 for prev, cur in zip(segs, segs[1:]) if cur.start < prev.end)
    assert overlaps > 0


def test_wav_length_covers_last_segment() -> None:
    utts = _utterances(5)
    wav, segs = simulate_meeting(utts, SR, overlap_ratio=0.2, rng=np.random.default_rng(5))
    assert len(wav) / SR == pytest.approx(max(s.end for s in segs), abs=1e-6)


def test_empty_input() -> None:
    wav, segs = simulate_meeting([], SR)
    assert wav.size == 0
    assert segs == []


def test_deterministic_with_seeded_rng() -> None:
    utts = _utterances(9)
    wav_a, segs_a = simulate_meeting(utts, SR, overlap_ratio=0.4, rng=np.random.default_rng(6))
    wav_b, segs_b = simulate_meeting(utts, SR, overlap_ratio=0.4, rng=np.random.default_rng(6))
    np.testing.assert_array_equal(wav_a, wav_b)
    assert segs_a == segs_b


def test_rejects_bad_overlap_ratio() -> None:
    with pytest.raises(ValueError):
        simulate_meeting(_utterances(2), SR, overlap_ratio=1.5)
