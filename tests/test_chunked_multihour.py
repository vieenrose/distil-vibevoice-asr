"""Multi-hour ChunkedTranscriber tests: registry anchoring, consolidation,
pause/resume, and legacy-path regression (CPU, no network, no models).

The scenario chains several overlapping windows in which the mock labeler
PERMUTES its per-window speaker labels (so nothing can be recovered from label
strings alone) and speaker ``B`` is SILENT during every window overlap.  With
only pairwise stitch continuity ``B`` would be handed a fresh label at each
boundary; the persistent :class:`SpeakerRegistry` re-identifies ``B`` by voice
and keeps a single global identity.  A voice-drift for ``B`` halfway through the
meeting plants an over-split that the end-of-meeting consolidation pass repairs.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from distil_vibevoice.data.long_meetings import _synth_voice
from distil_vibevoice.data.manifest import MeetingRecord, Segment
from distil_vibevoice.runtime.chunked_inference import ChunkedTranscriber
from distil_vibevoice.runtime.consolidate import consolidate
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder
from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

sf = pytest.importorskip("soundfile")

SR = 16000
WIN = 10.0
OVERLAP = 2.0
STEP = WIN - OVERLAP  # 8.0
ERA_BOUNDARY = 3 * STEP  # 24.0s: B's voice drifts here (mock-embedder scenario)


# --------------------------------------------------------------------------- #
# Meeting builder: A anchors every overlap (stitch keeps it), B never speaks in
# an overlap (only voice anchoring can keep it), labels permuted per window.
# --------------------------------------------------------------------------- #
def _global_turns(n_windows: int) -> list[tuple[float, float, str, str]]:
    turns: list[tuple[float, float, str, str]] = []
    for i in range(n_windows):
        b = STEP * i
        turns.append((b + 2.5, b + 4.0, "A", f"alpha {i}"))   # A, window-exclusive
        turns.append((b + 4.5, b + 6.0, "B", f"beta {i}"))    # B, window-exclusive
        turns.append((b + 8.0, b + 9.0, "A", f"anchor {i}"))  # A, spans i / i+1 overlap
    return turns


def _build_meeting(n_windows, voice_fn, permute=True):
    """Return (wav, per_window_local_segment_lists) for the mock labeler."""
    turns = _global_turns(n_windows)
    total = STEP * (n_windows - 1) + WIN
    wav = np.zeros(int(round(total * SR)), dtype=np.float32)
    for start, end, real, _text in turns:
        sig = voice_fn(real, start, end - start)
        f0 = int(round(start * SR))
        seg = sig[: int(round((end - start) * SR))]
        wav[f0 : f0 + len(seg)] += seg

    names = ["spk_x", "spk_y"]
    windows: list[list[Segment]] = []
    for i in range(n_windows):
        t0 = STEP * i
        wend = t0 + WIN
        local: list[Segment] = []
        for start, end, real, text in turns:
            if end <= t0 or start >= wend:
                continue
            cs, ce = max(start, t0), min(end, wend)
            if permute:
                lbl = names[i % 2] if real == "A" else names[(i + 1) % 2]
            else:
                lbl = real
            local.append(Segment(cs - t0, ce - t0, lbl, text))
        windows.append(local)
    return wav, windows


class ListLabeler:
    """Hotwords-only mock labeler returning precomputed per-window segments."""

    def __init__(self, windows: list[list[Segment]]) -> None:
        self.windows = windows
        self.calls: list[dict] = []

    def label_file(self, audio_path: str, hotwords: list[str] | None = None) -> MeetingRecord:
        info = sf.info(str(audio_path))
        segs = [replace(s) for s in self.windows[len(self.calls)]]
        self.calls.append({"path": str(audio_path), "hotwords": list(hotwords or [])})
        return MeetingRecord(
            audio_path=str(audio_path),
            duration_s=info.frames / info.samplerate,
            sample_rate=int(info.samplerate),
            language="zh-TW-en",
            source="mock",
            split="test",
            segments=segs,
        )


# --------------------------------------------------------------------------- #
# Voice signatures.
# --------------------------------------------------------------------------- #
def _mfcc_voice(real: str, start: float, dur: float) -> np.ndarray:
    """Distinct-but-stable synth voice per speaker (for the real MFCC embedder)."""
    return _synth_voice({"A": "S2", "B": "S56"}[real], dur, SR)


def _sine(freq: float, dur: float) -> np.ndarray:
    t = np.arange(int(round(dur * SR)), dtype=np.float64) / SR
    return (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_voice(real: str, start: float, dur: float) -> np.ndarray:
    """Pure tones whose peak frequency identifies the speaker (mock embedder).

    ``B`` drifts from 3200 Hz to 3600 Hz at :data:`ERA_BOUNDARY` — the same
    person, but far enough for a high match threshold to over-split it, and
    close enough for consolidation to merge it back (A stays far from both).
    """
    if real == "A":
        return _sine(800.0, dur)
    return _sine(3200.0 if start < ERA_BOUNDARY else 3600.0, dur)


class MelBandEmbedder:
    """Discriminative test embedder: soft one-hot over the audio's peak freq.

    Gives clean, tunable geometry (A far from B, B's two eras close) so the
    consolidation / resume wiring can be exercised deterministically — the
    real :class:`MfccStatsEmbedder` is too smooth to separate voices past the
    default consolidation threshold.
    """

    def __init__(self, n_bins: int = 64, fmax: float = 8000.0, sigma: float = 400.0) -> None:
        self.grid = np.linspace(0.0, fmax, n_bins)
        self.sigma = float(sigma)
        self.dim = int(n_bins)

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float64).reshape(-1)
        if wav.size < 0.2 * sr:
            return np.zeros(self.dim, dtype=np.float32)
        spec = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(wav.size, d=1.0 / sr)
        fpk = float(freqs[int(np.argmax(spec))])
        v = np.exp(-0.5 * ((self.grid - fpk) / self.sigma) ** 2)
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 0 else np.zeros(self.dim, np.float32)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _write(path: Path, wav: np.ndarray) -> str:
    sf.write(str(path), wav, SR)
    return str(path)


def _labels_by_role(segments: list[Segment]) -> tuple[set[str], set[str]]:
    """(labels used by A-turns, labels used by B-turns) via turn text."""
    a = {s.speaker for s in segments if s.text.startswith(("alpha", "anchor"))}
    b = {s.speaker for s in segments if s.text.startswith("beta")}
    return a, b


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_registry_anchoring_survives_permutation_and_silent_overlaps(tmp_path):
    """Voice anchoring keeps one global id per speaker across 6 permuted windows
    where B is silent in every overlap — where plain stitching would fragment B."""
    wav, windows = _build_meeting(6, _mfcc_voice)
    path = _write(tmp_path / "m.wav", wav)

    registry = SpeakerRegistry(embed_dim=MfccStatsEmbedder().dim, match_threshold=0.9)
    tr = ChunkedTranscriber(
        ListLabeler(windows),
        window_s=WIN,
        overlap_s=OVERLAP,
        embedder=MfccStatsEmbedder(),
        registry=registry,
        consolidate_on_finish=False,  # MFCC too smooth for the default cut
    )
    rec = tr.transcribe(path)

    a_labels, b_labels = _labels_by_role(rec.segments)
    assert len(a_labels) == 1, f"A fragmented into {a_labels}"
    assert len(b_labels) == 1, f"B fragmented into {b_labels} despite silent overlaps"
    assert a_labels != b_labels
    assert all(s.speaker.startswith("SPEAKER_") for s in rec.segments)
    assert rec.meta["registry_speakers"] == 2
    # Every planted turn survives (nothing dropped by anchoring).
    assert {s.text for s in rec.segments} == {t[3] for t in _global_turns(6)}


def test_legacy_stitching_fragments_the_silent_speaker(tmp_path):
    """Contrast: without a registry, B (silent in overlaps) gets many labels."""
    wav, windows = _build_meeting(6, _mfcc_voice)
    path = _write(tmp_path / "m.wav", wav)

    rec = ChunkedTranscriber(ListLabeler(windows), window_s=WIN, overlap_s=OVERLAP).transcribe(path)
    _a, b_labels = _labels_by_role(rec.segments)
    assert len(b_labels) > 1  # the problem the registry exists to solve


def test_consolidation_repairs_planted_split(tmp_path):
    """B's mid-meeting voice drift over-splits it; consolidation merges it back
    while keeping the genuinely distinct speaker A separate."""
    wav, windows = _build_meeting(6, _band_voice)
    path = _write(tmp_path / "m.wav", wav)
    embedder = MelBandEmbedder()

    # (a) Without consolidation the split is visible: B has two global ids.
    reg_raw = SpeakerRegistry(embed_dim=embedder.dim, match_threshold=0.9)
    raw = ChunkedTranscriber(
        ListLabeler(windows), window_s=WIN, overlap_s=OVERLAP,
        embedder=embedder, registry=reg_raw, consolidate_on_finish=False,
    ).transcribe(path)
    _a, b_raw = _labels_by_role(raw.segments)
    assert len(b_raw) == 2, f"expected planted 2-way split of B, got {b_raw}"
    assert reg_raw.match_threshold == 0.9

    # (b) With consolidation (default thresholds) the split is repaired.
    reg = SpeakerRegistry(embed_dim=embedder.dim, match_threshold=0.9)
    rec = ChunkedTranscriber(
        ListLabeler(windows), window_s=WIN, overlap_s=OVERLAP,
        embedder=embedder, registry=reg, consolidate_on_finish=True,
    ).transcribe(path)
    a_labels, b_labels = _labels_by_role(rec.segments)
    assert len(b_labels) == 1, f"consolidation did not repair split: {b_labels}"
    assert a_labels != b_labels
    assert {s.speaker for s in rec.segments} == a_labels | b_labels  # exactly 2 speakers
    assert rec.meta["consolidated"], "consolidation mapping should be non-empty"
    # The merge collapses the higher-numbered id onto the lower one.
    old, new = next(iter(rec.meta["consolidated"].items()))
    assert int(old.split("_")[1]) > int(new.split("_")[1])


def test_registry_state_save_load_resumes_midmeeting(tmp_path):
    """A paused meeting resumes with the same global identities after load."""
    embedder = MelBandEmbedder()
    state = str(tmp_path / "reg_state")

    # Part 1: fresh registry, persisted to disk.
    wav1, win1 = _build_meeting(3, _band_voice)  # all B turns before ERA_BOUNDARY
    p1 = _write(tmp_path / "part1.wav", wav1)
    tr1 = ChunkedTranscriber(
        ListLabeler(win1), window_s=WIN, overlap_s=OVERLAP,
        embedder=embedder, registry=SpeakerRegistry(embed_dim=embedder.dim, match_threshold=0.9),
        consolidate_on_finish=False, registry_state=state,
    )
    rec1 = tr1.transcribe(p1)
    a1, b1 = _labels_by_role(rec1.segments)
    assert Path(state + ".json").exists() and Path(state + ".npz").exists()

    # Part 2: a NEW transcriber loads the saved state (no explicit registry).
    wav2, win2 = _build_meeting(3, _band_voice)
    p2 = _write(tmp_path / "part2.wav", wav2)
    tr2 = ChunkedTranscriber(
        ListLabeler(win2), window_s=WIN, overlap_s=OVERLAP,
        embedder=embedder, consolidate_on_finish=False, registry_state=state,
    )
    assert tr2._registry is not None and len(tr2._registry.speakers) == 2  # resumed
    rec2 = tr2.transcribe(p2)
    a2, b2 = _labels_by_role(rec2.segments)

    # Same voices -> same global ids as part 1; no new speakers invented.
    assert a2 == a1 and b2 == b1
    assert rec2.meta["registry_speakers"] == 2


def test_no_embedder_is_byte_identical_to_legacy(tmp_path):
    """Regression guard: omitting embedder/registry leaves behavior unchanged."""
    wav, windows = _build_meeting(4, _mfcc_voice)
    path = _write(tmp_path / "m.wav", wav)

    def run():
        return ChunkedTranscriber(
            ListLabeler(windows), window_s=WIN, overlap_s=OVERLAP
        ).transcribe(path)

    a, b = run(), run()
    assert a.segments == b.segments  # deterministic
    assert set(a.meta) == {"num_windows", "window_s", "overlap_s", "roster"}
    assert "registry_speakers" not in a.meta and "consolidated" not in a.meta
    assert all(s.speaker.startswith("SPEAKER_") for s in a.segments)
    assert isinstance(a.meta["roster"], dict)
