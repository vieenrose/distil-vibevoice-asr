"""Concatenated-minimum-permutation word error rate (cpWER).

Segments are grouped per speaker into concatenated token streams (mixed
zh-char / en-word tokenization); the speaker permutation minimizing total
edit distance is found with the Hungarian algorithm.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

import numpy as np

from distil_vibevoice.eval.mer import levenshtein, tokenize_mixed

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime dep
    from distil_vibevoice.data.manifest import Segment

__all__ = ["cpwer", "speaker_token_streams", "optimal_speaker_map"]


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required for cpwer/der speaker assignment: pip install scipy"
        ) from exc
    return linear_sum_assignment(cost)


def speaker_token_streams(segments: Iterable["Segment"]) -> dict[str, list[str]]:
    """Concatenate each speaker's segments (in time order) into one token stream."""
    streams: dict[str, list[str]] = defaultdict(list)
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        streams[seg.speaker].extend(tokenize_mixed(seg.text))
    return dict(streams)


def _cost_matrix(ref_streams: list[list[str]], hyp_streams: list[list[str]]) -> np.ndarray:
    cost = np.zeros((len(ref_streams), len(hyp_streams)), dtype=np.int64)
    for i, r in enumerate(ref_streams):
        for j, h in enumerate(hyp_streams):
            cost[i, j] = levenshtein(r, h)
    return cost


def optimal_speaker_map(
    ref_segments: list["Segment"], hyp_segments: list["Segment"]
) -> dict[str, str]:
    """Hyp-speaker -> ref-speaker map minimizing total stream edit distance."""
    ref_streams = speaker_token_streams(ref_segments)
    hyp_streams = speaker_token_streams(hyp_segments)
    if not ref_streams or not hyp_streams:
        return {}
    ref_labels = list(ref_streams)
    hyp_labels = list(hyp_streams)
    cost = _cost_matrix(
        [ref_streams[r] for r in ref_labels], [hyp_streams[h] for h in hyp_labels]
    )
    rows, cols = _linear_sum_assignment(cost)
    return {hyp_labels[j]: ref_labels[i] for i, j in zip(rows, cols)}


def cpwer(ref_segments: list["Segment"], hyp_segments: list["Segment"]) -> float:
    """cpWER: min over speaker permutations of summed stream edit distance,
    divided by total reference tokens.

    Speaker sets of different sizes are padded with empty streams so that
    unmatched reference speech counts as deletions and unmatched hypothesis
    speech as insertions.  Edge cases: no ref tokens and no hyp tokens -> 0.0;
    no ref tokens but some hyp tokens -> 1.0.
    """
    ref_streams = list(speaker_token_streams(ref_segments).values())
    hyp_streams = list(speaker_token_streams(hyp_segments).values())
    total_ref = sum(len(s) for s in ref_streams)
    total_hyp = sum(len(s) for s in hyp_streams)
    if total_ref == 0:
        return 0.0 if total_hyp == 0 else 1.0
    n = max(len(ref_streams), len(hyp_streams))
    ref_streams += [[] for _ in range(n - len(ref_streams))]
    hyp_streams += [[] for _ in range(n - len(hyp_streams))]
    cost = _cost_matrix(ref_streams, hyp_streams)
    rows, cols = _linear_sum_assignment(cost)
    total_edits = int(cost[rows, cols].sum())
    return total_edits / total_ref
