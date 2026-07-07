"""Timestamp accuracy: MAE over greedily matched ref/hyp segments.

Segments are matched only when their speakers correspond under the optimal
(cpWER-style) speaker mapping AND their texts are similar (mixed-token Jaccard
> 0.5).  Matching is greedy by descending Jaccard, ties broken by temporal
proximity.  Unmatched segments are ignored by :func:`timestamp_mae` (per the
API contract it returns a single float) but are counted in the dict returned
by :func:`timestamp_report`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from distil_vibevoice.eval.cpwer import optimal_speaker_map
from distil_vibevoice.eval.mer import tokenize_mixed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.manifest import Segment

__all__ = ["timestamp_mae", "timestamp_report", "match_segments"]

_MIN_JACCARD = 0.5


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_segments(
    ref_segments: list["Segment"],
    hyp_segments: list["Segment"],
    min_jaccard: float = _MIN_JACCARD,
) -> list[tuple[int, int]]:
    """Greedy (ref_idx, hyp_idx) matching under speaker map + text similarity.

    A hyp segment may match a ref segment only if its speaker maps to the ref
    segment's speaker under the optimal speaker assignment and the mixed-token
    Jaccard similarity of the texts exceeds ``min_jaccard``.  Pairs are taken
    greedily by descending Jaccard (ties: smaller |start delta| first); each
    segment is used at most once.
    """
    if not ref_segments or not hyp_segments:
        return []
    speaker_map = optimal_speaker_map(ref_segments, hyp_segments)
    ref_tokens = [set(tokenize_mixed(s.text)) for s in ref_segments]
    hyp_tokens = [set(tokenize_mixed(s.text)) for s in hyp_segments]

    candidates: list[tuple[float, float, int, int]] = []
    for i, rseg in enumerate(ref_segments):
        for j, hseg in enumerate(hyp_segments):
            if speaker_map.get(hseg.speaker) != rseg.speaker:
                continue
            jac = _jaccard(ref_tokens[i], hyp_tokens[j])
            if jac > min_jaccard:
                candidates.append((-jac, abs(rseg.start - hseg.start), i, j))
    candidates.sort()

    used_ref: set[int] = set()
    used_hyp: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _neg_jac, _dt, i, j in candidates:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs


def timestamp_report(
    ref_segments: list["Segment"],
    hyp_segments: list["Segment"],
    min_jaccard: float = _MIN_JACCARD,
) -> dict:
    """Detailed timestamp accuracy report.

    Returns a dict with ``mae`` (mean absolute error over matched start AND
    end deltas, 0.0 when nothing matched), ``start_mae``, ``end_mae``,
    ``n_matched``, ``n_ref``, ``n_hyp``, ``n_ref_unmatched`` and
    ``n_hyp_unmatched``.
    """
    pairs = match_segments(ref_segments, hyp_segments, min_jaccard=min_jaccard)
    start_deltas = [abs(ref_segments[i].start - hyp_segments[j].start) for i, j in pairs]
    end_deltas = [abs(ref_segments[i].end - hyp_segments[j].end) for i, j in pairs]
    n = len(pairs)
    all_deltas = start_deltas + end_deltas
    return {
        "mae": sum(all_deltas) / len(all_deltas) if all_deltas else 0.0,
        "start_mae": sum(start_deltas) / n if n else 0.0,
        "end_mae": sum(end_deltas) / n if n else 0.0,
        "n_matched": n,
        "n_ref": len(ref_segments),
        "n_hyp": len(hyp_segments),
        "n_ref_unmatched": len(ref_segments) - n,
        "n_hyp_unmatched": len(hyp_segments) - n,
    }


def timestamp_mae(
    ref_segments: list["Segment"], hyp_segments: list["Segment"]
) -> float:
    """MAE (seconds) over start+end deltas of matched segments.

    Unmatched segments are ignored (see :func:`timestamp_report` for counts).
    Returns 0.0 when no segments match.
    """
    return float(timestamp_report(ref_segments, hyp_segments)["mae"])
