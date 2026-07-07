"""Tests for distil_vibevoice.runtime.diarize (CPU, numpy/scipy)."""
from __future__ import annotations

import numpy as np

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.runtime.diarize import SpeakerRegion, assign_text_to_speakers, diarize
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder

SR = 16000


def _voice(f0: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    return (sig + 0.03 * rng.standard_normal(t.size)).astype(np.float32)


def test_diarize_two_distinct_voices():
    rng = np.random.default_rng(0)
    a = _voice(110, 3.0, rng)
    b = _voice(260, 3.0, rng)
    wav = np.concatenate([a, b, a, b])
    regions = diarize(wav, SR, MfccStatsEmbedder(), win_s=1.0, hop_s=0.5, n_speakers=2, smooth=3)
    assert len({r.speaker for r in regions}) == 2
    assert regions == sorted(regions, key=lambda r: r.start)


def test_assign_text_by_overlap():
    regions = [
        SpeakerRegion(0.0, 5.0, "SPEAKER_0"),
        SpeakerRegion(5.0, 10.0, "SPEAKER_1"),
    ]
    segs = [Segment(1.0, 2.0, "X", "hi"), Segment(6.0, 7.0, "Y", "bye")]
    out = assign_text_to_speakers(segs, regions)
    assert [s.speaker for s in out] == ["SPEAKER_0", "SPEAKER_1"]
    assert [s.text for s in out] == ["hi", "bye"]


def test_diarize_silence_returns_single_region():
    wav = np.zeros(SR, dtype=np.float32)
    regions = diarize(wav, SR, MfccStatsEmbedder())
    assert len({r.speaker for r in regions}) <= 1
