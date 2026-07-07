"""Speaker embeddings for the long-form speaker registry.

Provides a small :class:`BaseEmbedder` protocol plus two implementations:

* :class:`MfccStatsEmbedder` — a dependency-light (numpy/scipy only) speaker
  embedder: 24 mel-cepstral coefficients (log-mel via :func:`scipy.signal.stft`
  followed by a DCT-II) summarized as per-coefficient mean, std, delta-mean and
  delta-std → a 96-dim L2-normalized float32 vector.  It is nowhere near a
  neural speaker model, but it separates clearly different voices and keeps the
  registry/consolidation pipeline testable on CPU without model downloads.
* an ONNX-backed embedder (via :func:`load_embedder` with ``kind='onnx'``) for
  real deployments, expecting an ECAPA-TDNN-style export (see
  :class:`OnnxSpeakerEmbedder`).

All embedders return an L2-normalized float32 vector of shape ``(dim,)``; too
short (or empty / silent) audio yields an all-zero vector so callers can treat
"no usable voice evidence" uniformly.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BaseEmbedder",
    "MfccStatsEmbedder",
    "OnnxSpeakerEmbedder",
    "EcapaEmbedder",
    "load_embedder",
]

#: Audio shorter than this (seconds) yields an all-zero embedding.
MIN_AUDIO_S = 0.2

#: Neural speaker models need a bit more context than the MFCC-stats embedder;
#: clips shorter than this (seconds) yield an all-zero embedding.
ECAPA_MIN_AUDIO_S = 0.3

#: Default speechbrain ECAPA-TDNN speaker-verification model (192-dim, VoxCeleb).
ECAPA_DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


#: Norms at or below this are treated as "no voice evidence" (-> zeros) rather
#: than amplified to a unit vector.  A silent/degenerate clip whose decorrelated
#: MFCC-stat blocks cancel leaves only floating-point roundoff (norm ~1e-15);
#: without this floor that roundoff would be blown up to a unit "garbage" vector
#: that spuriously matches every other silent clip at cosine ~1.0.
_MIN_NORM = 1e-8


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalized float32 copy of ``vec``; all-zero / sub-epsilon / non-finite -> zeros."""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < _MIN_NORM:
        return np.zeros(vec.shape, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _decorrelate_block(block: np.ndarray) -> np.ndarray:
    """Make an MFCC-statistic block reflect *speaker* shape, not shared level.

    The 0th cepstral coefficient (overall log-energy) and each block's DC offset
    are speaker-independent — they encode loudness / broad spectral level shared
    by all voiced speech — yet dominate the raw concatenated vector and inflate
    the cosine similarity between *different* speakers well above any usable
    match threshold.  Zeroing c0 and mean-removing the remaining coefficients
    (cepstral-mean-normalization in spirit) leaves the speaker-discriminative
    spectral *shape*, so distinct voices become near-orthogonal.  Length is
    preserved (so the public ``dim`` is unchanged).
    """
    block = np.array(block, dtype=np.float64).reshape(-1)
    if block.size == 0:
        return block
    block[0] = 0.0
    if block.size > 1:
        block[1:] -= block[1:].mean()
    return block


def _to_mono_f64(wav: np.ndarray) -> np.ndarray:
    """Flatten ``(T,)`` or soundfile-style ``(T, channels)`` audio to mono float64."""
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    elif wav.ndim != 1:
        raise ValueError(f"expected 1-D or (T, channels) audio, got shape {wav.shape}")
    return wav


@runtime_checkable
class BaseEmbedder(Protocol):
    """Anything that maps a waveform to a fixed-size L2-normalized voice vector."""

    dim: int

    def embed(self, wav: "np.ndarray", sr: int) -> "np.ndarray":
        """Return an L2-normalized float32 embedding of shape ``(dim,)``."""
        ...


class MfccStatsEmbedder:
    """Numpy/scipy-only speaker embedder based on MFCC statistics.

    Pipeline: hann-window STFT (25 ms frames, 10 ms hop) -> power spectrum ->
    ``n_mels`` triangular mel filters -> log -> DCT-II (ortho) keeping the
    first ``n_mfcc`` coefficients -> per-coefficient mean, std, delta-mean and
    delta-std over frames -> concatenate (``dim = 4 * n_mfcc``) -> L2
    normalize (float32).

    Robustness: audio shorter than :data:`MIN_AUDIO_S` seconds (or with an
    unusably low sample rate) returns ``np.zeros(dim, dtype=np.float32)``.
    """

    def __init__(
        self,
        n_mels: int = 40,
        n_mfcc: int = 24,
        frame_s: float = 0.025,
        hop_s: float = 0.010,
        fmin: float = 20.0,
    ) -> None:
        if n_mels < n_mfcc:
            raise ValueError(f"need n_mels >= n_mfcc, got {n_mels} < {n_mfcc}")
        if not 0 < hop_s <= frame_s:
            raise ValueError(f"need 0 < hop_s <= frame_s, got {hop_s} vs {frame_s}")
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.frame_s = float(frame_s)
        self.hop_s = float(hop_s)
        self.fmin = float(fmin)
        self.dim = 4 * self.n_mfcc

    # ------------------------------------------------------------------ #

    @staticmethod
    def _hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    @staticmethod
    def _mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    def _mel_filterbank(self, freqs: np.ndarray, sr: int) -> np.ndarray:
        """Triangular mel filterbank of shape ``(n_mels, len(freqs))``."""
        fmax = sr / 2.0
        mel_pts = np.linspace(
            self._hz_to_mel(self.fmin), self._hz_to_mel(fmax), self.n_mels + 2
        )
        hz_pts = self._mel_to_hz(mel_pts)
        fb = np.zeros((self.n_mels, freqs.size), dtype=np.float64)
        for i in range(self.n_mels):
            lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
            up = (freqs - lo) / max(mid - lo, 1e-12)
            down = (hi - freqs) / max(hi - mid, 1e-12)
            fb[i] = np.clip(np.minimum(up, down), 0.0, 1.0)
        return fb

    def _mfcc(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """MFCC matrix of shape ``(n_mfcc, n_frames)``."""
        from scipy.fft import dct
        from scipy.signal import stft

        nperseg = max(int(round(self.frame_s * sr)), 16)
        hop = max(int(round(self.hop_s * sr)), 1)
        freqs, _, spec = stft(
            wav,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg - hop,
            boundary=None,
            padded=False,
        )
        power = np.abs(spec) ** 2
        mel_power = self._mel_filterbank(freqs, sr) @ power
        log_mel = np.log(mel_power + 1e-10)
        return dct(log_mel, type=2, axis=0, norm="ortho")[: self.n_mfcc]

    # ------------------------------------------------------------------ #

    def embed(self, wav: "np.ndarray", sr: int) -> "np.ndarray":
        """Embed a waveform; deterministic, returns zeros for unusable audio."""
        zeros = np.zeros(self.dim, dtype=np.float32)
        if sr <= 0:
            return zeros
        wav = _to_mono_f64(wav)
        if wav.size < MIN_AUDIO_S * sr or sr / 2.0 <= self.fmin:
            return zeros
        # Degenerate (silent / constant) audio carries no voice evidence: its
        # decorrelated MFCC-stat blocks cancel to floating-point roundoff, so
        # short-circuit to zeros rather than let normalization amplify that
        # roundoff into a unit "garbage" vector that matches every other silent
        # clip.  (_l2_normalize's norm floor is the backstop for the same case.)
        if float(np.ptp(wav)) < 1e-9:
            return zeros
        mfcc = self._mfcc(wav, sr)
        if mfcc.shape[1] == 0:
            return zeros
        mean = mfcc.mean(axis=1)
        std = mfcc.std(axis=1)
        if mfcc.shape[1] >= 2:
            delta = np.diff(mfcc, axis=1)
            d_mean = delta.mean(axis=1)
            d_std = delta.std(axis=1)
        else:
            d_mean = np.zeros(self.n_mfcc, dtype=np.float64)
            d_std = np.zeros(self.n_mfcc, dtype=np.float64)
        blocks = [_decorrelate_block(b) for b in (mean, std, d_mean, d_std)]
        return _l2_normalize(np.concatenate(blocks))


class OnnxSpeakerEmbedder:
    """ONNX-backed speaker embedder (e.g. an exported ECAPA-TDNN).

    Expects a model exported to take a single float32 input of shape
    ``(1, T)`` — the raw 16 kHz waveform — and produce a ``(1, dim)`` (or
    ``(dim,)``) embedding, e.g. a speechbrain ECAPA-TDNN exported with::

        torch.onnx.export(
            model, torch.zeros(1, 16000), "ecapa.onnx",
            input_names=["wav"], output_names=["embedding"],
            dynamic_axes={"wav": {1: "T"}},
        )

    Audio at a different sample rate is resampled to 16 kHz
    (``scipy.signal.resample_poly``).  Output is L2-normalized float32; too
    short audio (< :data:`MIN_AUDIO_S` s) yields zeros without invoking the
    model.  Requires the optional ``onnxruntime`` dependency.
    """

    MODEL_SR = 16000

    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "onnxruntime is required for the 'onnx' speaker embedder: "
                "pip install onnxruntime"
            ) from e
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        out_shape = self._session.get_outputs()[0].shape
        last = out_shape[-1] if out_shape else None
        if not isinstance(last, int) or last < 1:
            raise ValueError(
                f"cannot infer embedding dim from ONNX output shape {out_shape!r}; "
                "expected a fixed last dimension (e.g. (1, 192) for ECAPA-TDNN)"
            )
        self.dim = int(last)

    def embed(self, wav: "np.ndarray", sr: int) -> "np.ndarray":
        if sr <= 0:
            return np.zeros(self.dim, dtype=np.float32)
        wav = _to_mono_f64(wav)
        if wav.size < MIN_AUDIO_S * sr:
            return np.zeros(self.dim, dtype=np.float32)
        if sr != self.MODEL_SR:
            from scipy.signal import resample_poly

            g = math.gcd(self.MODEL_SR, sr)
            wav = resample_poly(wav, self.MODEL_SR // g, sr // g)
        x = wav.astype(np.float32)[np.newaxis, :]
        (out,) = self._session.run(None, {self._input_name: x})
        return _l2_normalize(np.asarray(out))


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN speaker embedder (192-dim, VoxCeleb-trained).

    Wraps :class:`speechbrain.inference.speaker.EncoderClassifier` loaded from
    ``model_path`` (a local dir or HF repo id; defaults to
    :data:`ECAPA_DEFAULT_SOURCE`).  Heavy imports (torch / torchaudio /
    speechbrain) are deferred to construction so importing this module stays
    dependency-light.

    Incoming audio at any sample rate is resampled to 16 kHz
    (``scipy.signal.resample_poly``) inside :meth:`embed`, matching the model's
    training rate.  Output is an L2-normalized float32 vector of shape
    ``(192,)``; clips shorter than :data:`ECAPA_MIN_AUDIO_S` seconds (or with a
    non-positive sample rate) yield zeros without invoking the model.
    """

    MODEL_SR = 16000

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        savedir: str | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:  # pragma: no cover - optional heavy dependency
            raise ImportError(
                "speechbrain (and torch/torchaudio) are required for the 'ecapa' "
                "speaker embedder: pip install speechbrain torchaudio"
            ) from e
        source = str(model_path) if model_path is not None else ECAPA_DEFAULT_SOURCE
        if savedir is None:
            import os

            safe = source.replace("/", "--")
            savedir = os.path.join(
                os.path.expanduser("~/.cache/speechbrain"), safe
            )
        self._torch = __import__("torch")
        self._model = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": device}
        )
        self._model.eval()
        self.dim = 192

    def embed(self, wav: "np.ndarray", sr: int) -> "np.ndarray":
        zeros = np.zeros(self.dim, dtype=np.float32)
        if sr <= 0:
            return zeros
        wav = _to_mono_f64(wav)
        if wav.size < ECAPA_MIN_AUDIO_S * sr:
            return zeros
        if sr != self.MODEL_SR:
            from scipy.signal import resample_poly

            g = math.gcd(self.MODEL_SR, sr)
            wav = resample_poly(wav, self.MODEL_SR // g, sr // g)
        if wav.size < ECAPA_MIN_AUDIO_S * self.MODEL_SR:
            return zeros
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None, :]
        with torch.no_grad():
            emb = self._model.encode_batch(x)
        vec = emb.squeeze().detach().cpu().numpy().reshape(-1)
        if vec.shape[0] != self.dim:  # pragma: no cover - guards a wrong model
            raise ValueError(
                f"expected {self.dim}-dim ECAPA embedding, got {vec.shape[0]}"
            )
        return _l2_normalize(vec)


def load_embedder(kind: str = "mfcc", model_path: str | None = None) -> BaseEmbedder:
    """Instantiate a speaker embedder.

    ``kind='mfcc'`` returns the built-in :class:`MfccStatsEmbedder` (no extra
    dependencies).  ``kind='onnx'`` loads an ONNX speaker model from
    ``model_path`` via onnxruntime (lazy import; a clear ImportError is raised
    when onnxruntime is missing) — see :class:`OnnxSpeakerEmbedder` for the
    expected ECAPA-style export signature.  ``kind='ecapa'`` loads a SpeechBrain
    ECAPA-TDNN speaker-verification model (192-dim; ``model_path`` optional,
    defaulting to :data:`ECAPA_DEFAULT_SOURCE`) — see :class:`EcapaEmbedder`.
    """
    if kind == "mfcc":
        return MfccStatsEmbedder()
    if kind == "onnx":
        if model_path is None:
            raise ValueError("model_path is required for the 'onnx' embedder")
        return OnnxSpeakerEmbedder(model_path)
    if kind == "ecapa":
        return EcapaEmbedder(model_path)
    raise ValueError(
        f"unknown embedder kind {kind!r}; expected 'mfcc', 'onnx' or 'ecapa'"
    )
