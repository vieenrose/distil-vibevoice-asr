"""Global speaker-identity consistency for long-form meeting transcripts.

DER and cpWER both tolerate a *consistent* relabeling of speakers, so a system
that silently swaps two speakers' identities halfway through a multi-hour
meeting can still score perfectly on them.  :func:`speaker_consistency`
instead measures whether each hypothesis speaker keeps a single global identity
across the whole recording — exactly what the persistent speaker registry is
supposed to guarantee.

A single optimal hyp->ref speaker mapping is chosen for the entire meeting (a
time-overlap maximizing Hungarian assignment, in the same spirit as the
frame-overlap mapping in :mod:`distil_vibevoice.eval.der`); the score is the
fraction of hypothesis speaker-time whose mapped label matches a reference
speaker active at that instant.  1.0 means every hyp speaker's global label
lines up with its reference speaker everywhere; a block of one speaker's time
attributed to the wrong global id lowers the score in proportion to its
duration.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from distil_vibevoice.eval.cpwer import _linear_sum_assignment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.manifest import Segment

__all__ = ["speaker_consistency"]


def _activity_matrix(
    segments: list["Segment"], centers: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Boolean ``[n_speakers, n_frames]`` activity matrix and speaker order.

    A frame is active for a speaker when its center falls inside one of that
    speaker's segments, so overlapping speech is represented as multiple active
    rows in the same column.
    """
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


def speaker_consistency(
    ref_segments: list["Segment"],
    hyp_segments: list["Segment"],
    step: float = 0.01,
) -> float:
    """Fraction of hyp speaker-time whose global label matches the reference.

    A single hyp->ref speaker mapping is chosen to maximize total co-active
    frame overlap (Hungarian assignment); then every hyp speaker-frame is
    scored correct iff its mapped reference speaker is active in that frame.
    Speaker-time is counted per speaker, so a frame with two active hyp
    speakers contributes two units (matching the duration weighting used by
    the other metrics).

    Edge cases: no hyp speech (whether or not there is ref speech) is
    vacuously consistent -> 1.0; hyp speech against an empty reference has no
    identity to anchor to -> 0.0.  ``step`` is the frame size in seconds.
    """
    max_end = max(
        [seg.end for seg in ref_segments] + [seg.end for seg in hyp_segments],
        default=0.0,
    )
    if max_end <= 0.0:
        return 1.0
    n_frames = int(math.ceil(max_end / step))
    centers = (np.arange(n_frames, dtype=np.float64) + 0.5) * step

    ref_mat, _ = _activity_matrix(ref_segments, centers)
    hyp_mat, _ = _activity_matrix(hyp_segments, centers)

    total_hyp = int(hyp_mat.sum(dtype=np.int64))
    if total_hyp == 0:
        return 1.0
    if ref_mat.shape[0] == 0:
        return 0.0

    # Overlap[h, r] = frames where hyp speaker h and ref speaker r are both
    # active; maximize total overlap => minimize the negated cost.
    overlap = hyp_mat.astype(np.int64) @ ref_mat.T.astype(np.int64)
    rows, cols = _linear_sum_assignment(-overlap)
    mapping = {int(h): int(r) for h, r in zip(rows, cols)}

    correct = 0
    for h in range(hyp_mat.shape[0]):
        r = mapping.get(h)
        if r is not None:
            correct += int((hyp_mat[h] & ref_mat[r]).sum(dtype=np.int64))
    return correct / total_hyp
