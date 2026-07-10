"""Chain scripted meeting sections into hours-long recordings.

Multi-hour meetings (2-8h with coffee breaks) are the stress test for the
persistent speaker registry: the same people speak across many sections and
must keep a stable global identity.  :func:`build_long_meeting` concatenates
several :class:`~distil_vibevoice.data.dialogue_scripts.DialogueScript`
sections that share one speaker bank, inserting silence "breaks" between them,
and returns the full waveform plus exact per-turn :class:`Segment` labels.

Audio for each turn comes from ``tts_fn(text, ref_wav)``.  When no TTS is
available a deterministic per-speaker placeholder voice is synthesized: a
harmonic-plus-noise excitation shaped by a formant envelope whose parameters
are seeded purely from the speaker id.  This gives every speaker a stable
spectral signature that is identical across sections (so an embedder sees the
same voice again) yet distinct between speakers (so it can tell them apart),
without any model download.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Callable

import numpy as np

from distil_vibevoice.data.manifest import Segment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.dialogue_scripts import DialogueScript

__all__ = ["build_long_meeting"]

#: Seconds of synthetic audio per text character (placeholder voice).
SECONDS_PER_CHAR = 0.08
#: Floor on a turn's duration so even 1-2 character turns stay embeddable.
MIN_TURN_S = 0.30


def _speaker_seed(speaker: str) -> int:
    """Stable 64-bit seed from a speaker id (unlike the salted builtin hash)."""
    digest = hashlib.sha256(str(speaker).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


#: Number of mel bands used to design the placeholder spectral envelope.
_ENV_MELS = 40
#: Log-domain amplitude of the per-speaker cepstral signature (separation knob).
_ENV_AMP = 2.0


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _synth_voice(speaker: str, dur_s: float, sr: int) -> np.ndarray:
    """Deterministic placeholder voice for ``speaker`` of length ``dur_s``.

    The timbre depends only on ``speaker`` (via a stable hash seed), so the same
    speaker sounds identical in every section while different speakers get
    well-separated spectra; only the length depends on ``dur_s``.

    Design: each speaker gets a random *cepstrum* (MFCC coefficients 1..23) that
    is inverse-DCT'd into a log-mel envelope, then applied to a mostly-flat
    (noise + weak harmonic) excitation.  Because the excitation is spectrally
    flat, an MFCC-statistics embedder recovers approximately that random
    cepstrum, so two different speakers land at near-orthogonal embeddings while
    the same speaker (any duration) stays essentially identical — exactly the
    separation the speaker registry needs to keep distinct voices apart.
    """
    from scipy.fft import idct

    rng = np.random.default_rng(_speaker_seed(speaker))
    nyquist = sr * 0.5
    f0 = float(rng.uniform(90.0, 220.0))

    # Per-speaker cepstral signature -> log-mel envelope (c0 left at 0: level is
    # not a speaker cue, and the registry embedder ignores it anyway).
    n_ceps = min(24, _ENV_MELS)
    ceps = np.zeros(_ENV_MELS, dtype=np.float64)
    base = rng.standard_normal(n_ceps - 1)
    ceps[1:n_ceps] = base / (np.linalg.norm(base) + 1e-12) * _ENV_AMP * np.sqrt(n_ceps - 1)
    log_mel_env = idct(ceps, type=2, norm="ortho")

    n = max(int(round(dur_s * sr)), 1)
    t = np.arange(n, dtype=np.float64) / sr

    # Mostly-flat excitation: white noise (flat spectrum) + a weak harmonic
    # comb for a voiced character.  Flatness is what lets the embedder read off
    # the envelope cleanly.
    exc = rng.standard_normal(n)
    harm = np.zeros(n, dtype=np.float64)
    n_harm = max(int((0.45 * sr) // f0), 1)
    for k in range(1, n_harm + 1):
        harm += np.sin(2.0 * np.pi * k * f0 * t) / k
    exc = exc + 0.3 * harm / (np.std(harm) + 1e-12) * (np.std(exc) + 1e-12)

    # Apply the speaker envelope in the frequency domain.
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mel = _hz_to_mel(freqs)
    mel_edges = np.linspace(_hz_to_mel(20.0), _hz_to_mel(nyquist), _ENV_MELS)
    env = np.exp(np.interp(mel, mel_edges, log_mel_env))
    sig = np.fft.irfft(np.fft.rfft(exc) * env, n=n)

    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if peak > 0.0:
        sig = (sig / peak) * 0.9
    return sig.astype(np.float32)


def build_long_meeting(
    scripts: list["DialogueScript"],
    speaker_wavs: dict[str, "np.ndarray"],
    sr: int,
    tts_fn: "Callable[[str, np.ndarray], np.ndarray] | None" = None,
    break_range: tuple = (30.0, 120.0),
    rng=None,
) -> tuple["np.ndarray", list[Segment]]:
    """Concatenate meeting ``scripts`` into one long waveform + exact segments.

    Args:
        scripts: ordered meeting sections; all draw speakers from the shared
            ``speaker_wavs`` bank so identities recur across sections.
        speaker_wavs: per-speaker reference waveform, passed to ``tts_fn`` as
            the voice to clone.  With the placeholder synthesizer (``tts_fn is
            None``) its contents are unused — a speaker's voice is derived from
            its id — but the mapping still defines the roster.
        sr: output sample rate (shared by refs, TTS output and the mixture).
        tts_fn: ``tts_fn(text, ref_wav) -> mono float wav`` at ``sr``.  When
            None, a deterministic per-speaker placeholder voice is synthesized
            with duration ``~ len(text) * 0.08`` s.
        break_range: ``(lo, hi)`` seconds of silence inserted between adjacent
            sections (one break per gap; none before the first / after the
            last section).
        rng: numpy Generator for the break durations; fresh ``default_rng()``
            if None.  Voice synthesis is independent of it.

    Returns:
        ``(wav, segments)``: the float32 mixture and one :class:`Segment` per
        turn with absolute start/end times, in chronological order.  Every
        segment lies strictly within ``[0, len(wav) / sr]``; silence breaks
        appear as gaps between sections.
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = float(break_range[0]), float(break_range[1])
    if lo < 0.0 or hi < lo:
        raise ValueError(f"break_range must satisfy 0 <= lo <= hi, got {break_range!r}")

    chunks: list[np.ndarray] = []
    segments: list[Segment] = []
    cursor = 0  # samples written so far

    for si, script in enumerate(scripts):
        if si > 0:
            n_break = int(round(float(rng.uniform(lo, hi)) * sr))
            if n_break > 0:
                chunks.append(np.zeros(n_break, dtype=np.float32))
                cursor += n_break
        for turn in script.turns:
            if tts_fn is not None:
                ref = np.asarray(speaker_wavs[turn.speaker], dtype=np.float32)
                wav = np.asarray(tts_fn(turn.text, ref), dtype=np.float32).reshape(-1)
            else:
                dur = max(len(turn.text) * SECONDS_PER_CHAR, MIN_TURN_S)
                wav = _synth_voice(turn.speaker, dur, sr)
            if wav.size == 0:
                continue
            chunks.append(wav)
            segments.append(
                Segment(
                    start=cursor / sr,
                    end=(cursor + wav.size) / sr,
                    speaker=turn.speaker,
                    text=turn.text,
                )
            )
            cursor += wav.size

    if not chunks:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(chunks).astype(np.float32), segments
