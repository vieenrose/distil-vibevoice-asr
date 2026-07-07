"""End-of-meeting speaker consolidation.

Windowed anchoring (:mod:`distil_vibevoice.runtime.speaker_registry`) can
occasionally split one real speaker across two global ids (a bad boundary, a
noisy window).  :func:`consolidate` runs a constrained agglomerative clustering
over the per-speaker mean *segment* embeddings accumulated in the registry and
retroactively merges global ids that clearly belong to the same voice, leaving
distinct voices untouched.

Average-linkage agglomerative clustering on cosine distance (scipy); the
lowest-numbered ``SPEAKER_{n}`` in a cluster wins as the canonical id.  Fully
deterministic.  Pure numpy/scipy — no torch.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.manifest import Segment
    from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

__all__ = ["consolidate", "recluster_segments"]

_ID_RE = re.compile(r"^SPEAKER_(\d+)$")


def _id_num(global_id: str) -> float:
    m = _ID_RE.match(global_id)
    return float(int(m.group(1))) if m else math.inf


def _canonical(ids: list[str]) -> str:
    """Lowest-numbered ``SPEAKER_{n}`` (ties broken lexicographically)."""
    return min(ids, key=lambda g: (_id_num(g), g))


def _mean_embeddings(
    registry: "SpeakerRegistry",
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-speaker L2-normalized mean segment embedding and segment count."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for gid, _s, _e, emb in registry.segment_store:
        emb = np.asarray(emb, dtype=np.float64).reshape(-1)
        if gid in sums:
            sums[gid] += emb
        else:
            sums[gid] = emb.copy()
        counts[gid] = counts.get(gid, 0) + 1

    means: dict[str, np.ndarray] = {}
    for gid, total in sums.items():
        norm = float(np.linalg.norm(total))
        if norm > 0.0 and math.isfinite(norm):
            means[gid] = (total / norm).astype(np.float64)
    return means, counts


def _cluster(reliable: list[str], means: dict[str, np.ndarray], threshold: float) -> dict[str, str]:
    """Merge reliable speakers whose mean embeddings cluster within threshold."""
    if len(reliable) < 2:
        return {gid: gid for gid in reliable}

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    feats = np.stack([means[gid] for gid in reliable])
    dist = pdist(feats, metric="cosine")
    link = linkage(dist, method="average")
    labels = fcluster(link, t=threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for gid, lab in zip(reliable, labels):
        clusters.setdefault(int(lab), []).append(gid)

    mapping: dict[str, str] = {}
    for members in clusters.values():
        canon = _canonical(members)
        for gid in members:
            mapping[gid] = canon
    return mapping


def recluster_segments(
    registry: "SpeakerRegistry",
    segments: list["Segment"],
    distance_threshold: float = 0.7,
) -> tuple[list["Segment"], dict[str, str]]:
    """Global per-segment reclustering — the robust multi-window diarization pass.

    Incremental registry matching (first-match-wins + EMA) is fragile: once it
    wrongly merges two real speakers it cannot split them again, and
    :func:`consolidate` (which clusters per-speaker *means*) can only merge
    further.  This pass instead clusters the RAW per-segment embeddings accumulated
    in ``registry.segment_store`` (average-linkage agglomerative, cosine distance,
    cut at ``distance_threshold``) and relabels every segment by its cluster,
    decoupling the final diarization from the incremental assignment.

    On real ECAPA embeddings this recovers ~0.86 speaker-consistency on
    teacher-hypothesized segments vs ~0.55 for the incremental registry (see
    docs/LONG_MEETING_EVAL.md).  ``distance_threshold`` must match the embedder's
    separation: ~0.7 for ECAPA-TDNN (cross-speaker cosine ~0.14); a lower value
    for weaker embedders.  Marker segments (``[Noise]``/``[Silence]``, never
    embedded) keep their label.  Deterministic.

    Returns ``(relabeled_segments_copy, old_to_new_mapping)`` where the mapping is
    the majority new-label per old id (informational).
    """
    store = list(registry.segment_store)
    if len(store) < 2:
        return list(segments), {}

    embs = np.stack([np.asarray(e, dtype=np.float64).reshape(-1) for _g, _s, _e, e in store])

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    labels = fcluster(linkage(pdist(embs, metric="cosine"), method="average"),
                      t=distance_threshold, criterion="distance")

    # (start,end) -> cluster label, and old-id -> label votes.
    key2lab: dict[tuple[float, float], int] = {}
    old_votes: dict[str, dict[int, int]] = {}
    for (gid, s, e, _emb), lab in zip(store, labels):
        key2lab[(round(float(s), 3), round(float(e), 3))] = int(lab)
        old_votes.setdefault(gid, {})[int(lab)] = old_votes.setdefault(gid, {}).get(int(lab), 0) + 1

    # Canonical id per cluster, assigned in order of first appearance for stability.
    lab2canon: dict[int, str] = {}
    relabeled: list["Segment"] = []
    for seg in segments:
        lab = key2lab.get((round(float(seg.start), 3), round(float(seg.end), 3)))
        if lab is None:
            relabeled.append(seg)  # marker / unmatched — keep label
            continue
        if lab not in lab2canon:
            lab2canon[lab] = f"SPEAKER_{len(lab2canon)}"
        relabeled.append(replace(seg, speaker=lab2canon[lab]))

    mapping = {
        gid: lab2canon[max(votes, key=votes.get)]
        for gid, votes in old_votes.items()
        if votes and max(votes, key=votes.get) in lab2canon
    }
    return relabeled, mapping


def consolidate(
    registry: "SpeakerRegistry",
    segments: list["Segment"],
    distance_threshold: float = 0.45,
    min_segments: int = 3,
    mode: str = "merge",
    recluster_threshold: float = 0.7,
) -> tuple[list["Segment"], dict[str, str]]:
    """Merge global speaker ids that share a voice; relabel a copy of segments.

    Speakers with at least ``min_segments`` stored segment embeddings form the
    clustering basis (average-linkage agglomerative clustering over cosine
    distance between per-speaker mean embeddings, cut at
    ``distance_threshold``).  Speakers with fewer segments keep their label
    unless their mean embedding lands within ``distance_threshold`` of a
    reliable cluster — a "clean" merge.  The lowest-numbered id in each cluster
    wins.

    Returns ``(relabeled_segments_copy, old_to_new_mapping)`` and applies the
    same mapping to the registry via :meth:`SpeakerRegistry.relabel`.  The
    mapping only contains ids that actually change.  Deterministic.

    ``mode="recluster"`` instead delegates to :func:`recluster_segments` (global
    per-segment clustering at ``recluster_threshold``) — the robust path for
    multi-window diarization with a discriminative embedder (ECAPA).  Default
    ``mode="merge"`` preserves the per-speaker-mean merge behavior.
    """
    if mode == "recluster":
        return recluster_segments(registry, segments, distance_threshold=recluster_threshold)

    means, counts = _mean_embeddings(registry)

    reliable = sorted(
        (gid for gid in means if counts.get(gid, 0) >= min_segments), key=_id_num
    )
    unreliable = sorted(
        (gid for gid in means if counts.get(gid, 0) < min_segments), key=_id_num
    )

    mapping = _cluster(reliable, means, distance_threshold)

    # Canonical representative embedding per reliable cluster.
    canon_embs: dict[str, np.ndarray] = {}
    for gid, canon in mapping.items():
        if canon not in canon_embs:
            canon_embs[canon] = means[canon]

    # Attach an under-supported speaker only if it merges cleanly.
    for gid in unreliable:
        best_canon: str | None = None
        best_dist = distance_threshold
        for canon in sorted(canon_embs, key=_id_num):
            dist = 1.0 - float(np.dot(means[gid], canon_embs[canon]))
            if dist <= best_dist:
                # <= keeps the lowest-numbered canon on ties (sorted order).
                if best_canon is None or dist < best_dist:
                    best_canon, best_dist = canon, dist
        mapping[gid] = best_canon if best_canon is not None else gid

    # Keep only entries that actually change a label.
    mapping = {old: new for old, new in mapping.items() if old != new}

    registry.relabel(mapping)
    relabeled = [
        replace(seg, speaker=mapping.get(seg.speaker, seg.speaker)) for seg in segments
    ]
    return relabeled, mapping
