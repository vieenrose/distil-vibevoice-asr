"""Tests for distil_vibevoice.runtime.embeddings (CPU, numpy/scipy only)."""
from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.runtime.embeddings import (
    BaseEmbedder,
    MfccStatsEmbedder,
    load_embedder,
)

SR = 16000


def _synth_voice(f0: float, formant: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    """A crude but deterministic 'voice': harmonics of f0 shaped by a formant.

    Different (f0, formant) pairs give clearly different mel spectra; the same
    pair with fresh noise / duration gives near-identical MFCC statistics.
    """
    t = np.arange(int(SR * dur)) / SR
    ks = np.arange(1, 26)
    env = np.exp(-((ks * f0 - formant) ** 2) / (2 * 300.0 ** 2))
    sig = np.zeros_like(t)
    for k in ks:
        sig += env[k - 1] * np.sin(2 * np.pi * f0 * k * t) / k
    sig += 0.05 * rng.standard_normal(t.size)
    return sig.astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class TestMfccStatsEmbedder:
    def test_shape_and_normalization(self):
        emb = MfccStatsEmbedder()
        assert emb.dim == 96
        rng = np.random.default_rng(0)
        vec = emb.embed(_synth_voice(150, 800, 1.0, rng), SR)
        assert vec.shape == (96,)
        assert vec.dtype == np.float32
        assert _cos(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_discriminates_two_voices(self):
        emb = MfccStatsEmbedder()
        rng = np.random.default_rng(0)
        a = [emb.embed(_synth_voice(120, 700, d, rng), SR) for d in (1.5, 1.1, 1.3)]
        b = [emb.embed(_synth_voice(240, 1600, d, rng), SR) for d in (1.5, 1.2, 1.4)]

        same = min(_cos(a[0], a[1]), _cos(a[0], a[2]), _cos(b[0], b[1]))
        cross = max(_cos(a[0], b[0]), _cos(a[1], b[1]), _cos(a[2], b[2]))
        # Same-voice pairs are strictly more similar than cross-voice pairs.
        assert same > cross + 0.02

    def test_short_and_empty_audio_returns_zeros(self):
        emb = MfccStatsEmbedder()
        for wav in (np.zeros(10, dtype=np.float32), np.array([], dtype=np.float32)):
            vec = emb.embed(wav, SR)
            assert vec.shape == (96,)
            assert vec.dtype == np.float32
            assert np.all(vec == 0.0)

    def test_long_silent_audio_returns_zeros(self):
        # Regression: a long-enough all-zero clip used to normalize the
        # decorrelated MFCC-stat blocks' floating-point roundoff into a unit
        # "garbage" vector, so every silent clip matched every other at cosine
        # ~1.0.  It must instead be inert (all-zero "no voice evidence").
        emb = MfccStatsEmbedder()
        for dur in (0.5, 2.0, 3.0):
            vec = emb.embed(np.zeros(int(SR * dur), dtype=np.float32), SR)
            assert vec.shape == (96,)
            assert vec.dtype == np.float32
            assert np.all(vec == 0.0), f"silent {dur}s should embed to zeros"

    def test_two_silent_clips_are_not_similar(self):
        # Two unrelated silent regions must not collapse into one identity: with
        # all-zero embeddings their cosine is 0, not 1.
        emb = MfccStatsEmbedder()
        a = emb.embed(np.zeros(int(SR * 2.0), dtype=np.float32), SR)
        b = emb.embed(np.zeros(int(SR * 3.0), dtype=np.float32), SR)
        assert _cos(a, b) == 0.0

    def test_constant_nonzero_audio_returns_zeros(self):
        # Constant (DC) audio is equally degenerate: no voice evidence -> zeros.
        emb = MfccStatsEmbedder()
        vec = emb.embed(np.full(int(SR * 2.0), 0.3, dtype=np.float32), SR)
        assert np.all(vec == 0.0)

    def test_zero_sample_rate_returns_zeros(self):
        emb = MfccStatsEmbedder()
        rng = np.random.default_rng(0)
        vec = emb.embed(_synth_voice(150, 800, 1.0, rng), 0)
        assert np.all(vec == 0.0)

    def test_deterministic(self):
        emb = MfccStatsEmbedder()
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        v1 = emb.embed(_synth_voice(180, 900, 1.2, rng1), SR)
        v2 = emb.embed(_synth_voice(180, 900, 1.2, rng2), SR)
        assert np.array_equal(v1, v2)

    def test_stereo_is_averaged_to_mono(self):
        emb = MfccStatsEmbedder()
        rng = np.random.default_rng(0)
        mono = _synth_voice(150, 800, 1.0, rng)
        stereo = np.stack([mono, mono], axis=1)
        assert np.allclose(emb.embed(mono, SR), emb.embed(stereo, SR), atol=1e-6)


class TestLoadEmbedder:
    def test_default_is_mfcc(self):
        emb = load_embedder()
        assert isinstance(emb, MfccStatsEmbedder)
        assert isinstance(emb, BaseEmbedder)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown embedder kind"):
            load_embedder("bogus")

    def test_onnx_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path is required"):
            load_embedder("onnx")


def _ecapa_available() -> bool:
    """True iff speechbrain imports and the ECAPA weights are cached locally."""
    try:
        import speechbrain  # noqa: F401
    except Exception:
        return False
    from pathlib import Path

    from distil_vibevoice.runtime.embeddings import ECAPA_DEFAULT_SOURCE

    safe = ECAPA_DEFAULT_SOURCE.replace("/", "--")
    savedir = Path.home() / ".cache" / "speechbrain" / safe
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe}"
    return savedir.exists() or hub.exists()


class TestEcapaEmbedder:
    def test_ecapa_unknown_still_rejected(self):
        with pytest.raises(ValueError, match="unknown embedder kind"):
            load_embedder("bogus")

    @pytest.mark.skipif(
        not _ecapa_available(),
        reason="speechbrain / ECAPA-TDNN weights not available",
    )
    def test_ecapa_shape_and_normalization(self):
        emb = load_embedder("ecapa")
        assert emb.dim == 192
        assert isinstance(emb, BaseEmbedder)
        rng = np.random.default_rng(0)
        vec = emb.embed(_synth_voice(150, 800, 1.0, rng), SR)
        assert vec.shape == (192,)
        assert vec.dtype == np.float32
        assert _cos(vec, vec) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.skipif(
        not _ecapa_available(),
        reason="speechbrain / ECAPA-TDNN weights not available",
    )
    def test_ecapa_short_and_bad_sr_returns_zeros(self):
        emb = load_embedder("ecapa")
        rng = np.random.default_rng(0)
        # < ECAPA_MIN_AUDIO_S (0.3 s) at 16 kHz.
        short = _synth_voice(150, 800, 0.1, rng)
        assert np.all(emb.embed(short, SR) == 0.0)
        assert np.all(emb.embed(_synth_voice(150, 800, 1.0, rng), 0) == 0.0)
