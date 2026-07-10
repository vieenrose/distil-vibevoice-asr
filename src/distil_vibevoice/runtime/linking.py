"""Cross-window speaker linking: core-segment AHC + nearest-centroid assign.

Validated on real IVOD meetings (scripts/34b + sweep, 2026-07): per-segment
ECAPA embeddings, but only segments >= MIN_CORE_DUR_S participate in the
agglomerative clustering (short segments have unreliable embeddings and
fragment the clusters); the remaining segments are assigned to the nearest
cluster centroid. Average linkage, cosine distance, threshold 0.45.

Results vs pyannote reference (300 s windows):
  30-min meeting  : consistency 0.891, DER 0.056, 7 speakers (ref 8)
  123-min meeting : consistency 0.912, DER 0.180, 35 clusters (ref 26)
The old per-segment-all @0.7 config under-clustered badly on real far-field
audio (DER 0.474 on the 30-min meeting).
"""
from __future__ import annotations

import numpy as np

MIN_CORE_DUR_S = 3.0
AHC_THRESHOLD = 0.45


def link_speakers(
    segs: list[dict],
    embs: dict[int, np.ndarray],
    threshold: float = AHC_THRESHOLD,
    min_core_dur_s: float = MIN_CORE_DUR_S,
) -> list[str]:
    """Return one global speaker label ("S01", ...) per segment.

    ``segs`` need ``start``/``end`` keys; ``embs`` maps segment index ->
    L2-normalized embedding (segments without a usable embedding inherit the
    previous segment's label).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    core = [i for i in embs
            if segs[i]["end"] - segs[i]["start"] >= min_core_dur_s]
    lab_of: dict[int, int] = {}
    if len(core) >= 2:
        X = np.stack([embs[i] for i in core])
        labs = fcluster(linkage(pdist(X, "cosine"), "average"),
                        t=threshold, criterion="distance")
        lab_of = {i: int(l) for i, l in zip(core, labs)}
        cents: dict[int, list[np.ndarray]] = {}
        for i, l in lab_of.items():
            cents.setdefault(l, []).append(embs[i])
        keys = list(cents)
        M = np.stack([np.mean(cents[k], axis=0) for k in keys])
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        for i in embs:
            if i not in lab_of:
                lab_of[i] = keys[int(np.argmax(M @ embs[i]))]
    elif embs:  # too few long segments to cluster: everyone is one speaker
        lab_of = {i: 1 for i in embs}

    canon: dict[int, str] = {}
    out: list[str] = []
    prev = "S01"
    for i in range(len(segs)):
        l = lab_of.get(i)
        if l is None:
            spk = prev
        else:
            if l not in canon:
                canon[l] = f"S{len(canon) + 1:02d}"
            spk = canon[l]
        prev = spk
        out.append(spk)
    return out
