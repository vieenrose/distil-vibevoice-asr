"""Waveform augmentation for synthetic meeting audio (numpy/scipy only).

Chain: optional RIR convolution -> optional MUSAN noise mixing at a random
SNR -> probabilistic codec simulation (downsample round-trip + mu-law
quantization) -> random gain jitter. Every stage is skippable (dir=None),
the output is clip-safe float32 with the input's shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["augment_wav"]

_AUDIO_EXTS = (".wav",)


def _list_wavs(root: str) -> list[Path]:
    """Recursively list .wav files under ``root`` (sorted for determinism)."""
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in _AUDIO_EXTS)


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav file as mono float32 in [-1, 1] using scipy only."""
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        x = x.astype(np.float32)
    return x, int(sr)


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling via scipy."""
    from math import gcd

    from scipy.signal import resample_poly

    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def _apply_rir(wav: np.ndarray, sr: int, rir_dir: str, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a random room impulse response, preserving length/level."""
    from scipy.signal import fftconvolve

    rirs = _list_wavs(rir_dir)
    if not rirs:
        return wav
    rir, rir_sr = _load_wav_mono(rirs[int(rng.integers(len(rirs)))])
    if rir_sr != sr:
        rir = _resample(rir, rir_sr, sr)
    if rir.size == 0 or not np.any(np.abs(rir) > 0):
        return wav
    # Align on the direct path (max tap) so speech onset timing is preserved.
    peak = int(np.argmax(np.abs(rir)))
    wet = fftconvolve(wav, rir, mode="full")[peak : peak + wav.shape[0]]
    if wet.shape[0] < wav.shape[0]:
        wet = np.pad(wet, (0, wav.shape[0] - wet.shape[0]))
    # Renormalize to the dry signal's RMS.
    wet = wet * (_rms(wav) / _rms(wet))
    return wet.astype(np.float32)


def _mix_noise(
    wav: np.ndarray,
    sr: int,
    musan_dir: str,
    snr_db_range: tuple,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix a random MUSAN clip at a uniform-random SNR from ``snr_db_range``."""
    noises = _list_wavs(musan_dir)
    if not noises:
        return wav
    noise, noise_sr = _load_wav_mono(noises[int(rng.integers(len(noises)))])
    if noise_sr != sr:
        noise = _resample(noise, noise_sr, sr)
    if noise.size == 0 or _rms(noise) < 1e-8:
        return wav
    n = wav.shape[0]
    if noise.shape[0] < n:
        noise = np.tile(noise, int(np.ceil(n / noise.shape[0])))
    start = int(rng.integers(0, noise.shape[0] - n + 1))
    noise = noise[start : start + n]
    snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    gain = _rms(wav) / (_rms(noise) * (10.0 ** (snr_db / 20.0)))
    return (wav + gain * noise).astype(np.float32)


def _codec_sim(wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a lossy telephony/VoIP codec: narrowband round-trip + mu-law."""
    low_sr = int(rng.choice([8000, 16000]))
    if low_sr < sr:
        x = _resample(wav, sr, low_sr)
        x = _resample(x, low_sr, sr)
        # Round-trip may drift by a sample or two; fix the length.
        if x.shape[0] >= wav.shape[0]:
            x = x[: wav.shape[0]]
        else:
            x = np.pad(x, (0, wav.shape[0] - x.shape[0]))
    else:
        x = wav
    # Mild mu-law companding quantization (255 levels).
    mu = 255.0
    peak = float(np.max(np.abs(x)) + 1e-12)
    y = x / peak
    comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    comp = np.round(comp * mu) / mu
    y = np.sign(comp) * (np.expm1(np.abs(comp) * np.log1p(mu))) / mu
    return (y * peak).astype(np.float32)


def augment_wav(
    wav: "np.ndarray",
    sr: int,
    rir_dir: str | None = None,
    musan_dir: str | None = None,
    codec_prob: float = 0.3,
    snr_db_range: tuple = (5.0, 25.0),
    rng: "np.random.Generator|None" = None,
) -> "np.ndarray":
    """Augment a mono waveform for meeting-audio realism.

    Args:
        wav: mono float waveform, shape (T,).
        sr: sample rate of ``wav``.
        rir_dir: directory of RIR .wav files (searched recursively); None skips.
        musan_dir: directory of noise .wav files; None skips.
        codec_prob: probability of applying codec simulation.
        snr_db_range: (lo, hi) uniform SNR range in dB for noise mixing.
        rng: numpy Generator for determinism; a fresh default_rng() if None.

    Returns:
        float32 waveform with the same shape, values clipped to [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"expected mono waveform of shape (T,), got {x.shape}")
    orig_len = x.shape[0]

    if rir_dir is not None:
        x = _apply_rir(x, sr, rir_dir, rng)
    if musan_dir is not None:
        x = _mix_noise(x, sr, musan_dir, snr_db_range, rng)
    if codec_prob > 0 and rng.random() < codec_prob:
        x = _codec_sim(x, sr, rng)

    # Random gain jitter of +/- 3 dB.
    gain_db = float(rng.uniform(-3.0, 3.0))
    x = x * (10.0 ** (gain_db / 20.0))

    # Clip-safe: soft-rescale if we exceed full scale, then hard clip.
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    x = np.clip(x, -1.0, 1.0)

    assert x.shape[0] == orig_len
    return x.astype(np.float32)
