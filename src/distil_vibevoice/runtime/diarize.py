"""Decoupled speaker diarization — independent of the ASR teacher's boundaries.

The distilled ASR teacher's own diarization caps multi-window speaker-consistency
(~0.86) because its segment boundaries are noisy (short segments, turn-spanning
segments).  This module diarizes the raw audio *directly* — slide short windows
over speech, ECAPA-embed each, cluster with temporal smoothing — then attaches the
teacher's transcript text to the resulting speaker regions by timestamp overlap.
This removes the teacher-boundary ceiling and does not depend on the teacher for
"who spoke", so it also survives audio where the teacher's own diarization fails.

Pure numpy/scipy + a pluggable embedder; no torch here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from distil_vibevoice.data.manifest import Segment
    from distil_vibevoice.runtime.embeddings import BaseEmbedder

__all__ = ["diarize", "assign_text_to_speakers", "SpeakerRegion"]


@dataclass
class SpeakerRegion:
    start: float
    end: float
    speaker: str


def _speech_mask(wav: np.ndarray, sr: int, frame_s: float = 0.03, thresh_db: float = -35.0) -> np.ndarray:
    """Energy VAD: bool per frame (frame_s hop)."""
    n = int(frame_s * sr)
    if n < 1:
        n = 1
    nfr = len(wav) // n
    if nfr == 0:
        return np.zeros(0, dtype=bool)
    e = (wav[: nfr * n].reshape(nfr, n) ** 2).mean(1) + 1e-10
    ref = np.percentile(e, 95)
    return 10 * np.log10(e / (ref + 1e-10)) > thresh_db


def _windows(wav: np.ndarray, sr: int, speech: np.ndarray, win_s: float, hop_s: float,
             frame_s: float) -> list[tuple[float, float]]:
    """Window spans (s) whose centre frame is speech."""
    dur = len(wav) / sr
    out: list[tuple[float, float]] = []
    t = 0.0
    while t + win_s <= dur + hop_s:
        end = min(t + win_s, dur)
        fi = int((t + win_s / 2) / frame_s)
        if 0 <= fi < len(speech) and speech[fi]:
            out.append((t, end))
        t += hop_s
    return out


def _estimate_k(feats: np.ndarray, max_k: int = 10) -> int:
    """Speaker count via the largest eigengap of the normalized affinity."""
    from scipy.spatial.distance import pdist, squareform

    if len(feats) < 3:
        return max(1, len(feats))
    aff = 1.0 - squareform(pdist(feats, metric="cosine"))
    np.fill_diagonal(aff, 1.0)
    d = aff.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    lap = np.eye(len(aff)) - (dinv[:, None] * aff * dinv[None, :])
    ev = np.sort(np.linalg.eigvalsh(lap))
    upper = min(max_k, len(ev) - 1)
    if upper < 2:
        return 1
    gaps = np.diff(ev[: upper + 1])
    return int(np.argmax(gaps[1:]) + 2)  # skip the trivial first gap


def diarize(
    wav: np.ndarray,
    sr: int,
    embedder: "BaseEmbedder",
    win_s: float = 2.0,
    hop_s: float = 0.75,
    frame_s: float = 0.03,
    n_speakers: int | None = None,
    smooth: int = 3,
) -> list[SpeakerRegion]:
    """Diarize raw audio into contiguous speaker regions.

    Slides ``win_s`` windows (``hop_s`` hop) over VAD speech, ECAPA-embeds each,
    estimates the speaker count via eigengap (unless ``n_speakers`` given),
    clusters (agglomerative, cosine), median-smooths the label sequence
    (``smooth`` windows) to enforce temporal continuity, and merges adjacent
    same-speaker windows.  Deterministic.
    """
    wav = np.asarray(wav, dtype=np.float32)
    speech = _speech_mask(wav, sr, frame_s)
    spans = _windows(wav, sr, speech, win_s, hop_s, frame_s)
    if len(spans) < 2:
        dur = len(wav) / sr
        return [SpeakerRegion(0.0, dur, "SPEAKER_0")] if dur > 0 else []

    feats = np.stack([embedder.embed(wav[int(a * sr):int(b * sr)], sr) for a, b in spans])

    k = n_speakers if n_speakers else _estimate_k(feats)
    k = max(1, min(k, len(spans)))
    if k == 1:
        labels = np.zeros(len(spans), dtype=int)
    else:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        labels = fcluster(linkage(pdist(feats, "cosine"), "average"), t=k, criterion="maxclust")

    # temporal median smoothing over the ordered window sequence
    if smooth > 1 and len(labels) >= smooth:
        sm = labels.copy()
        h = smooth // 2
        for i in range(len(labels)):
            lo, hi = max(0, i - h), min(len(labels), i + h + 1)
            vals, cnt = np.unique(labels[lo:hi], return_counts=True)
            sm[i] = vals[np.argmax(cnt)]
        labels = sm

    # canonical ids in order of first appearance, merge adjacent same-speaker
    canon: dict[int, str] = {}
    regions: list[SpeakerRegion] = []
    for (a, b), lab in zip(spans, labels):
        lab = int(lab)
        if lab not in canon:
            canon[lab] = f"SPEAKER_{len(canon)}"
        sid = canon[lab]
        if regions and regions[-1].speaker == sid and a <= regions[-1].end + hop_s:
            regions[-1].end = max(regions[-1].end, b)
        else:
            regions.append(SpeakerRegion(a, b, sid))
    return regions


def assign_text_to_speakers(
    text_segments: list["Segment"],
    regions: list[SpeakerRegion],
) -> list["Segment"]:
    """Relabel each ASR text segment by the speaker region it overlaps most."""
    from dataclasses import replace

    out: list["Segment"] = []
    for seg in text_segments:
        best, best_ov = seg.speaker, 0.0
        for r in regions:
            ov = min(seg.end, r.end) - max(seg.start, r.start)
            if ov > best_ov:
                best, best_ov = r.speaker, ov
        out.append(replace(seg, speaker=best))
    return out
