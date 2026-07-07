"""Tests for distil_vibevoice.data.augment (CPU-only, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.data.augment import augment_wav

SR = 24000


def _sine(freq: float = 440.0, dur_s: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path, wav: np.ndarray, sr: int = SR) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), sr, (np.clip(wav, -1, 1) * 32767).astype(np.int16))


@pytest.fixture()
def rir_dir(tmp_path):
    d = tmp_path / "rirs"
    d.mkdir()
    # Simple exponentially-decaying impulse response.
    rng = np.random.default_rng(0)
    ir = rng.standard_normal(2400).astype(np.float32) * np.exp(-np.arange(2400) / 300.0)
    ir[0] = 1.0
    _write_wav(d / "room0.wav", ir / np.max(np.abs(ir)))
    return str(d)


@pytest.fixture()
def musan_dir(tmp_path):
    d = tmp_path / "musan"
    (d / "noise").mkdir(parents=True)
    rng = np.random.default_rng(1)
    _write_wav(d / "noise" / "n0.wav", 0.3 * rng.standard_normal(SR // 2).astype(np.float32))
    return str(d)


def test_noop_dirs_none_preserves_shape_dtype() -> None:
    wav = _sine()
    out = augment_wav(wav, SR, rir_dir=None, musan_dir=None, codec_prob=0.0,
                      rng=np.random.default_rng(0))
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_full_chain(rir_dir, musan_dir) -> None:
    wav = _sine(dur_s=0.8)
    out = augment_wav(wav, SR, rir_dir=rir_dir, musan_dir=musan_dir, codec_prob=1.0,
                      snr_db_range=(5.0, 15.0), rng=np.random.default_rng(7))
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
    # The chain must actually change the signal.
    assert not np.allclose(out, wav)


def test_deterministic_with_seeded_rng(rir_dir, musan_dir) -> None:
    wav = _sine(dur_s=0.5)
    a = augment_wav(wav, SR, rir_dir=rir_dir, musan_dir=musan_dir, codec_prob=1.0,
                    rng=np.random.default_rng(42))
    b = augment_wav(wav, SR, rir_dir=rir_dir, musan_dir=musan_dir, codec_prob=1.0,
                    rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_noise_mixing_changes_signal(musan_dir) -> None:
    wav = _sine(dur_s=0.5)
    out = augment_wav(wav, SR, musan_dir=musan_dir, codec_prob=0.0,
                      snr_db_range=(0.0, 5.0), rng=np.random.default_rng(3))
    assert not np.allclose(out, wav)


def test_codec_only() -> None:
    wav = _sine(dur_s=0.4)
    out = augment_wav(wav, SR, codec_prob=1.0, rng=np.random.default_rng(9))
    assert out.shape == wav.shape
    assert np.all(np.isfinite(out))


def test_empty_dirs_are_skipped(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    wav = _sine(dur_s=0.3)
    out = augment_wav(wav, SR, rir_dir=str(empty), musan_dir=str(empty),
                      codec_prob=0.0, rng=np.random.default_rng(0))
    assert out.shape == wav.shape
    assert np.all(np.isfinite(out))


def test_rejects_non_mono() -> None:
    with pytest.raises(ValueError):
        augment_wav(np.zeros((2, 100), dtype=np.float32), SR,
                    rng=np.random.default_rng(0))
