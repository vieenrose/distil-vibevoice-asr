"""Frame-based diarization error rate (DER) with collar and overlap handling.

Time is discretized into fixed frames (default 10 ms); each frame carries the
*set* of active speakers for ref and hyp, so overlapping speech is scored
correctly.  A collar excludes frames within +-collar of every reference
segment boundary.  The hyp->ref speaker mapping is the one maximizing total
frame overlap (Hungarian assignment).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from distil_vibevoice.eval.cpwer import _linear_sum_assignment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.manifest import Segment

__all__ = ["der"]


def _activity_matrix(
    segments: list["Segment"], centers: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Boolean [n_speakers, n_frames] activity matrix and speaker order."""
    speakers: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        if seg.speaker not in seen:
            seen.add(seg.speaker)
            speakers.append(seg.speaker)
    mat = np.zeros((len(speakers), centers.size), dtype=bool)
    index = {spk: i for i, spk in enumerate(speakers)}
    for seg in segments:
        if seg.end > seg.start:
            mat[index[seg.speaker]] |= (centers >= seg.start) & (centers < seg.end)
    return mat, speakers


def der(
    ref_segments: list["Segment"],
    hyp_segments: list["Segment"],
    collar: float = 0.25,
    step: float = 0.01,
) -> float:
    """DER = (missed + false alarm + speaker confusion) / scored ref speech.

    All quantities are counted in speaker-frames, so a frame with two missed
    reference speakers contributes two misses.  Frames within +-collar of any
    reference segment boundary are excluded from scoring.  Edge cases: no
    scored reference speech -> 0.0 if the hypothesis is also silent there,
    else 1.0.
    """
    max_end = max(
        [seg.end for seg in ref_segments] + [seg.end for seg in hyp_segments],
        default=0.0,
    )
    if max_end <= 0.0:
        return 0.0
    n_frames = int(math.ceil(max_end / step))
    centers = (np.arange(n_frames, dtype=np.float64) + 0.5) * step

    ref_mat, _ = _activity_matrix(ref_segments, centers)
    hyp_mat, _ = _activity_matrix(hyp_segments, centers)

    scored = np.ones(n_frames, dtype=bool)
    if collar > 0.0:
        for seg in ref_segments:
            for boundary in (seg.start, seg.end):
                scored &= np.abs(centers - boundary) >= collar

    ref_s = ref_mat[:, scored]
    hyp_s = hyp_mat[:, scored]

    # Optimal hyp->ref speaker mapping by scored frame overlap.
    mapped_hyp = np.zeros_like(ref_s)
    if ref_s.shape[0] and hyp_s.shape[0]:
        overlap = ref_s.astype(np.int64) @ hyp_s.T.astype(np.int64)
        rows, cols = _linear_sum_assignment(-overlap)
        for r, h in zip(rows, cols):
            mapped_hyp[r] = hyp_s[h]

    n_ref = ref_s.sum(axis=0, dtype=np.int64)
    n_hyp = hyp_s.sum(axis=0, dtype=np.int64)
    n_correct = (ref_s & mapped_hyp).sum(axis=0, dtype=np.int64)

    miss = int(np.maximum(n_ref - n_hyp, 0).sum())
    false_alarm = int(np.maximum(n_hyp - n_ref, 0).sum())
    confusion = int(np.maximum(np.minimum(n_ref, n_hyp) - n_correct, 0).sum())

    denom = int(n_ref.sum())
    if denom == 0:
        return 0.0 if int(n_hyp.sum()) == 0 else 1.0
    return (miss + false_alarm + confusion) / denom
