"""Audio deduplication against eval sets via a spectral fingerprint.

The fingerprint is a sha1 over a coarsely quantized log-mel-band energy
contour, computed with numpy + scipy only (no external services):

1. downmix to mono, peak-normalize (gain-invariant),
2. magnitude STFT with a 50 ms rectangular hop (20 frames/s),
3. 16 triangular mel-spaced band filters over 50 Hz .. Nyquist,
4. average-pool band energies to 2 frames per second,
5. log10, then 3-bit quantization against the contour's own min/max,
6. sha1 hexdigest of the quantized byte matrix.

Identical audio always maps to the same fingerprint; any audible edit
(added noise, different content, resampling artifacts) is expected to
change it. This is an exact-match dedupe key, not a fuzzy matcher.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from distil_vibevoice.data.manifest import MeetingRecord, iter_manifest

logger = logging.getLogger(__name__)

__all__ = ["audio_fingerprint", "build_eval_index", "filter_against_index"]

_N_BANDS = 16
_FPS = 2.0  # fingerprint frames per second
_HOP_S = 0.05  # STFT hop (seconds); 0.5 s pooling blocks = 10 STFT frames
_FMIN_HZ = 50.0
_QUANT_LEVELS = 8  # 3-bit


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(n_bands: int, n_fft_bins: int, sr: int) -> np.ndarray:
    """Triangular mel-spaced filters, shape (n_bands, n_fft_bins)."""
    fmax = sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(_FMIN_HZ), _hz_to_mel(fmax), n_bands + 2)
    hz_pts = np.asarray(_mel_to_hz(mel_pts), dtype=np.float64)
    freqs = np.linspace(0.0, fmax, n_fft_bins)
    fb = np.zeros((n_bands, n_fft_bins), dtype=np.float64)
    for b in range(n_bands):
        lo, mid, hi = hz_pts[b], hz_pts[b + 1], hz_pts[b + 2]
        rising = (freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs) / max(hi - mid, 1e-9)
        fb[b] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def audio_fingerprint(wav: np.ndarray, sr: int) -> str:
    """Deterministic spectral hash of an audio signal (hex sha1 string)."""
    from scipy.signal import stft

    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim == 2:
        # Downmix along the channel axis; soundfile yields (frames, channels).
        wav = wav.mean(axis=1 if wav.shape[0] >= wav.shape[1] else 0)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.0:
        wav = wav / peak

    nperseg = max(int(round(sr * _HOP_S)), 16)
    if wav.size < nperseg:
        wav = np.pad(wav, (0, nperseg - wav.size))
    _, _, spec = stft(wav, fs=sr, nperseg=nperseg, noverlap=0, padded=True)
    power = np.abs(spec) ** 2  # (bins, frames)

    fb = _mel_filterbank(_N_BANDS, power.shape[0], sr)
    bands = fb @ power  # (n_bands, frames)

    # Pool STFT frames into fingerprint frames of 1/_FPS seconds each.
    frames_per_block = max(int(round(1.0 / (_FPS * _HOP_S))), 1)
    n_blocks = max(bands.shape[1] // frames_per_block, 1)
    trimmed = bands[:, : n_blocks * frames_per_block]
    pooled = trimmed.reshape(_N_BANDS, n_blocks, -1).mean(axis=2)

    contour = np.log10(pooled + 1e-10)
    lo, hi = float(contour.min()), float(contour.max())
    scaled = (contour - lo) / (hi - lo + 1e-12)
    quant = np.clip((scaled * _QUANT_LEVELS).astype(np.int64), 0, _QUANT_LEVELS - 1)

    h = hashlib.sha1()
    h.update(np.asarray(quant.shape, dtype=np.int64).tobytes())
    h.update(quant.astype(np.uint8).tobytes())
    return h.hexdigest()


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to fingerprint audio files: pip install soundfile"
        ) from e
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav, sr


def build_eval_index(manifests: list[str]) -> set[str]:
    """Fingerprint every audio file referenced by the given eval manifests.

    Missing audio files are skipped with a logged warning.
    """
    index: set[str] = set()
    for manifest in manifests:
        for rec in iter_manifest(manifest):
            path = Path(rec.audio_path)
            if not path.exists():
                logger.warning("build_eval_index: missing audio %s; skipping", path)
                continue
            wav, sr = _read_audio(path)
            index.add(audio_fingerprint(wav, sr))
    return index


def filter_against_index(
    records: Iterable[MeetingRecord], index: set[str], audio_root: str = ""
) -> list[MeetingRecord]:
    """Drop records whose audio fingerprint collides with the eval index.

    ``audio_root`` is prepended to relative audio paths. Records whose audio
    file is missing cannot be checked and are KEPT with a logged warning.
    """
    kept: list[MeetingRecord] = []
    for rec in records:
        path = Path(audio_root) / rec.audio_path if audio_root else Path(rec.audio_path)
        if not path.exists():
            logger.warning(
                "filter_against_index: missing audio %s; keeping record unchecked", path
            )
            kept.append(rec)
            continue
        wav, sr = _read_audio(path)
        if audio_fingerprint(wav, sr) in index:
            logger.info("filter_against_index: dropping eval-duplicate %s", path)
            continue
        kept.append(rec)
    return kept
