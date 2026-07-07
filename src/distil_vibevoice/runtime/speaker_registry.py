"""Persistent global-speaker registry for multi-hour meeting transcription.

Long recordings are transcribed window by window; each window's speakers are
*local* to that window.  Rather than chaining pairwise stitches (where one bad
boundary poisons every subsequent window), the :class:`SpeakerRegistry`
anchors every window-local speaker to a *global* identity via voice-embedding
centroids (EMA-updated) plus short text snippets — a star topology, so a single
mismatch never cascades.

State is fully serializable (JSON metadata + an ``.npz`` array sidecar) so a
transcription can be paused and resumed, and a segment-level embedding store is
kept for the end-of-meeting consolidation pass
(:mod:`distil_vibevoice.runtime.consolidate`) that retroactively fixes
historical stitch errors.

Pure numpy — no torch, no model downloads.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SpeakerProfile", "SpeakerRegistry"]

#: Maximum number of text snippets retained per speaker profile.
_MAX_SNIPPETS = 3
#: Global speaker id format.
_ID_RE = re.compile(r"^SPEAKER_(\d+)$")


#: Norms at or below this are treated as all-zero rather than amplified: an EMA
#: sum or weighted-merge of near-opposite centroids can cancel to floating-point
#: roundoff (norm ~1e-15), which must stay inert instead of being blown up into a
#: unit "garbage" centroid that spuriously matches everything.
_MIN_NORM = 1e-8


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalized float32 copy; all-zero / sub-epsilon / non-finite input -> zeros."""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < _MIN_NORM:
        return np.zeros(vec.shape, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _id_num(global_id: str) -> int:
    """Trailing integer of a ``SPEAKER_{n}`` label (inf if it does not match)."""
    m = _ID_RE.match(global_id)
    return int(m.group(1)) if m else math.inf  # type: ignore[return-value]


@dataclass
class SpeakerProfile:
    """A single global speaker: its running centroid, snippets and activity."""

    global_id: str
    centroid: np.ndarray  # (embed_dim,), L2-normalized float32
    n_updates: int
    snippets: list[str] = field(default_factory=list)
    last_active: float = 0.0


class SpeakerRegistry:
    """Anchor window-local speakers to persistent global identities.

    Parameters
    ----------
    embed_dim:
        Dimensionality of the voice embeddings fed in.
    ema:
        Centroid exponential-moving-average weight on the *old* centroid; each
        :meth:`update` sets ``centroid <- normalize(ema*old + (1-ema)*emb)``.
    match_threshold:
        Minimum cosine similarity for :meth:`match` to anchor to an existing
        speaker; below it, ``match`` returns ``(None, best_score)`` and the
        caller should allocate a :meth:`new_speaker`.
    """

    def __init__(
        self, embed_dim: int, ema: float = 0.9, match_threshold: float = 0.60
    ) -> None:
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if not 0.0 <= ema < 1.0:
            raise ValueError(f"need 0 <= ema < 1, got {ema}")
        self.embed_dim = int(embed_dim)
        self.ema = float(ema)
        self.match_threshold = float(match_threshold)
        self.speakers: dict[str, SpeakerProfile] = {}
        self.segment_store: list[tuple[str, float, float, np.ndarray]] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #

    def _coerce_emb(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size != self.embed_dim:
            raise ValueError(
                f"embedding dim {emb.size} != registry embed_dim {self.embed_dim}"
            )
        return emb

    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        """Best cosine match against existing centroids.

        Returns ``(global_id, score)`` when the best cosine similarity is
        ``>= match_threshold``; otherwise ``(None, best_score)``.  An empty
        registry (or all-zero embedding) yields ``(None, 0.0)``.
        """
        emb = self._coerce_emb(emb)
        query = _l2_normalize(emb)
        best_id: str | None = None
        best_score = 0.0
        # Deterministic tie-breaking: iterate speakers in id-number order.
        for gid in sorted(self.speakers, key=_id_num):
            score = float(np.dot(query, self.speakers[gid].centroid))
            if best_id is None or score > best_score:
                best_id, best_score = gid, score
        if best_id is not None and best_score >= self.match_threshold:
            return best_id, best_score
        return None, best_score

    def update(
        self, global_id: str, emb: np.ndarray, snippet: str, t_end: float
    ) -> None:
        """EMA-update a speaker's centroid, append a snippet and touch activity."""
        emb = self._coerce_emb(emb)
        prof = self.speakers.get(global_id)
        if prof is None:
            raise KeyError(f"unknown speaker {global_id!r}; call new_speaker first")
        prof.centroid = _l2_normalize(self.ema * prof.centroid + (1.0 - self.ema) * emb)
        prof.n_updates += 1
        if snippet:
            prof.snippets.append(snippet)
            if len(prof.snippets) > _MAX_SNIPPETS:
                prof.snippets = prof.snippets[-_MAX_SNIPPETS:]
        prof.last_active = float(t_end)

    def new_speaker(self, emb: np.ndarray, snippet: str, t_end: float) -> str:
        """Allocate a fresh ``SPEAKER_{n}`` seeded with ``emb``; return its id."""
        emb = self._coerce_emb(emb)
        gid = f"SPEAKER_{self._next_id}"
        self._next_id += 1
        self.speakers[gid] = SpeakerProfile(
            global_id=gid,
            centroid=_l2_normalize(emb),
            n_updates=1,
            snippets=[snippet] if snippet else [],
            last_active=float(t_end),
        )
        return gid

    def add_segment(
        self, global_id: str, start: float, end: float, emb: np.ndarray
    ) -> None:
        """Append a segment embedding to the store used by consolidation."""
        emb = self._coerce_emb(emb)
        self.segment_store.append((global_id, float(start), float(end), emb.copy()))

    def roster_prompt(self, max_speakers: int = 12, now: float | None = None) -> str:
        """Most-recently-active speakers, one ``'<id>: <snippet>'`` line each.

        Ordered by descending ``last_active`` (ties broken by id number),
        capped at ``max_speakers``.  ``now`` is accepted for API symmetry with
        the transcriber's clock and does not change the ordering.
        """
        if max_speakers < 1:
            return ""
        ordered = sorted(
            self.speakers.values(), key=lambda p: (-p.last_active, _id_num(p.global_id))
        )
        lines = []
        for prof in ordered[:max_speakers]:
            snippet = prof.snippets[-1] if prof.snippets else ""
            lines.append(f"{prof.global_id}: {snippet}")
        return "\n".join(lines)

    def relabel(self, mapping: dict[str, str]) -> None:
        """Apply a consolidation ``old_id -> new_id`` map to profiles and store.

        Ids absent from ``mapping`` are left unchanged.  When several old ids
        collapse onto the same new id their profiles are merged: centroids are
        combined weighted by ``n_updates`` (then renormalized), update counts
        summed, snippets concatenated (most recent kept) and ``last_active``
        maxed.  The segment store is relabeled in place.
        """
        resolve = lambda gid: mapping.get(gid, gid)

        merged: dict[str, SpeakerProfile] = {}
        # Merge in id-number order so the lowest-numbered source seeds ordering.
        for gid in sorted(self.speakers, key=_id_num):
            prof = self.speakers[gid]
            new_id = resolve(gid)
            tgt = merged.get(new_id)
            if tgt is None:
                merged[new_id] = SpeakerProfile(
                    global_id=new_id,
                    centroid=(prof.n_updates * prof.centroid).astype(np.float64),
                    n_updates=prof.n_updates,
                    snippets=list(prof.snippets),
                    last_active=prof.last_active,
                )
            else:
                tgt.centroid = tgt.centroid + prof.n_updates * prof.centroid
                tgt.n_updates += prof.n_updates
                tgt.snippets = (tgt.snippets + list(prof.snippets))[-_MAX_SNIPPETS:]
                tgt.last_active = max(tgt.last_active, prof.last_active)

        for prof in merged.values():
            prof.centroid = _l2_normalize(prof.centroid)
        self.speakers = merged
        self.segment_store = [
            (resolve(gid), s, e, emb) for gid, s, e, emb in self.segment_store
        ]

    # ------------------------------------------------------------------ #

    @staticmethod
    def _paths(path: str | Path) -> tuple[Path, Path]:
        base = Path(path)
        return base.with_suffix(".json"), base.with_suffix(".npz")

    def save(self, path: str | Path) -> None:
        """Serialize to ``<path>.json`` (metadata) + ``<path>.npz`` (arrays).

        Centroids and segment embeddings are stored in the ``.npz`` sidecar so
        the round-trip is bitwise-exact.
        """
        json_path, npz_path = self._paths(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        speakers_meta = []
        for i, gid in enumerate(self.speakers):
            prof = self.speakers[gid]
            key = f"centroid_{i}"
            arrays[key] = np.asarray(prof.centroid, dtype=np.float32)
            speakers_meta.append(
                {
                    "global_id": prof.global_id,
                    "n_updates": int(prof.n_updates),
                    "snippets": list(prof.snippets),
                    "last_active": float(prof.last_active),
                    "centroid_key": key,
                }
            )

        segments_meta = []
        for j, (gid, start, end, emb) in enumerate(self.segment_store):
            key = f"seg_{j}"
            arrays[key] = np.asarray(emb, dtype=np.float32)
            segments_meta.append(
                {
                    "global_id": gid,
                    "start": float(start),
                    "end": float(end),
                    "emb_key": key,
                }
            )

        meta = {
            "embed_dim": self.embed_dim,
            "ema": self.ema,
            "match_threshold": self.match_threshold,
            "next_id": self._next_id,
            "speakers": speakers_meta,
            "segments": segments_meta,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # Always write the sidecar (empty arrays dict is fine) for a stable pair.
        np.savez(npz_path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerRegistry":
        """Load a registry previously written by :meth:`save`."""
        json_path, npz_path = cls._paths(path)
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        with np.load(npz_path) as npz:
            arrays = {k: npz[k] for k in npz.files}

        reg = cls(
            embed_dim=int(meta["embed_dim"]),
            ema=float(meta["ema"]),
            match_threshold=float(meta["match_threshold"]),
        )
        reg._next_id = int(meta["next_id"])
        for sm in meta["speakers"]:
            reg.speakers[sm["global_id"]] = SpeakerProfile(
                global_id=sm["global_id"],
                centroid=np.asarray(arrays[sm["centroid_key"]], dtype=np.float32),
                n_updates=int(sm["n_updates"]),
                snippets=list(sm["snippets"]),
                last_active=float(sm["last_active"]),
            )
        for gm in meta["segments"]:
            reg.segment_store.append(
                (
                    gm["global_id"],
                    float(gm["start"]),
                    float(gm["end"]),
                    np.asarray(arrays[gm["emb_key"]], dtype=np.float32),
                )
            )
        return reg
