"""Cross-chunk speaker permutation matching and transcript stitching.

Long-form audio is transcribed in overlapping windows; each window's speaker
labels are local to that window.  :func:`match_speakers` aligns a new window's
speakers to the accumulated global roster using token agreement inside the
overlap region (Hungarian assignment), and :func:`stitch` applies the maps
chunk by chunk, de-duplicates segments in the overlaps and merges adjacent
same-speaker segments.

All segment times are GLOBAL (absolute seconds from the start of the full
recording); the caller is responsible for offsetting window-local times.
"""
from __future__ import annotations

import re
from dataclasses import replace

from distil_vibevoice.data.manifest import Segment

try:  # eval package may be incomplete while modules are brought up independently
    from distil_vibevoice.eval.mer import tokenize_mixed
except ImportError:  # pragma: no cover - fallback mirrors eval.mer semantics
    import unicodedata

    _TOKEN_RE = re.compile(
        "[一-鿿㐀-䶿〇\uf900-\ufaff\U00020000-\U0003134f]"
        "|[a-zà-öø-ɏ]+(?:['’][a-zà-öø-ɏ]+)*"
        "|[0-9]+"
    )

    def tokenize_mixed(text: str) -> list[str]:
        """Mixed zh/en tokens: one CJK char, latin word, or digit run per token."""
        return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


__all__ = ["match_speakers", "stitch"]

#: Adjacent same-speaker segments closer than this gap (seconds) are merged.
MERGE_GAP_S = 0.3
#: Minimum agreement score for a Hungarian pair to count as a match.
_MIN_AGREEMENT = 1e-9
#: Intersection / shorter-duration above which two segments are duplicate
#: candidates.  Containment (not IoU) so a window-boundary-clipped fragment of
#: a turn is still recognized as a duplicate of the full transcription.
_DUP_CONTAINMENT = 0.8
#: Token-jaccard above which differently-labeled overlapping segments are duplicates.
_DUP_TEXT_JACCARD = 0.5
#: Token containment (shared / shorter set) treating a clipped fragment's text
#: as a duplicate of the full text even when the jaccard is low.
_DUP_TEXT_CONTAINMENT = 0.8
#: A duplicate is kept over the other when it is this much longer (the
#: un-clipped version of a boundary-straddling turn wins).
_KEEP_LONGER_RATIO = 1.25

_GLOBAL_ID_RE = re.compile(r"^SPEAKER_(\d+)$")
_CJK_RE = re.compile("[一-鿿㐀-䶿〇\uf900-\ufaff\U00020000-\U0003134f]")


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _time_intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total intersection duration (seconds) between two interval lists."""
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def _speaker_profiles(
    segments: list[Segment], lo: float, hi: float
) -> dict[str, tuple[set[str], list[tuple[float, float]]]]:
    """Per-speaker (token set, clipped intervals) for segments intersecting [lo, hi]."""
    profiles: dict[str, tuple[set[str], list[tuple[float, float]]]] = {}
    for seg in segments:
        if seg.end <= lo or seg.start >= hi:
            continue
        tokens, intervals = profiles.setdefault(seg.speaker, (set(), []))
        tokens.update(tokenize_mixed(seg.text))
        intervals.append((max(seg.start, lo), min(seg.end, hi)))
    return profiles


def _ordered_speakers(segments: list[Segment]) -> list[str]:
    """Unique speaker labels ordered by first appearance (segment start time)."""
    seen: dict[str, None] = {}
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        seen.setdefault(seg.speaker, None)
    return list(seen)


def _next_free_global_id(used: set[str]) -> str:
    """Smallest 'SPEAKER_{n}' label not present in ``used``."""
    n = 0
    for label in used:
        m = _GLOBAL_ID_RE.match(label)
        if m:
            n = max(n, int(m.group(1)) + 1)
    while f"SPEAKER_{n}" in used:
        n += 1
    return f"SPEAKER_{n}"


def match_speakers(
    prev_segments: list[Segment],
    new_segments: list[Segment],
    overlap_start: float,
    overlap_end: float,
) -> dict[str, str]:
    """Map new-chunk speaker labels to previous-chunk (global) labels.

    For every (prev_speaker, new_speaker) pair the agreement score is the
    token-jaccard of what both said inside ``[overlap_start, overlap_end]``,
    weighted by the time-intersection (seconds) of their speech in the
    overlap.  The optimal one-to-one map is found with the Hungarian
    algorithm; new speakers with no positive-agreement match get fresh global
    ids ``'SPEAKER_{n}'`` (numbered after the ids already in use).

    Returns a dict covering EVERY speaker label present in ``new_segments``.
    """
    prev_profiles = _speaker_profiles(prev_segments, overlap_start, overlap_end)
    new_profiles = _speaker_profiles(new_segments, overlap_start, overlap_end)
    new_speakers = _ordered_speakers(new_segments)

    mapping: dict[str, str] = {}
    prev_ids = list(prev_profiles)
    cand_ids = [s for s in new_speakers if s in new_profiles]
    if prev_ids and cand_ids:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        score = np.zeros((len(prev_ids), len(cand_ids)), dtype=np.float64)
        for i, p in enumerate(prev_ids):
            p_tokens, p_intervals = prev_profiles[p]
            for j, q in enumerate(cand_ids):
                q_tokens, q_intervals = new_profiles[q]
                score[i, j] = _jaccard(p_tokens, q_tokens) * _time_intersection(
                    p_intervals, q_intervals
                )
        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            if score[r, c] > _MIN_AGREEMENT:
                mapping[cand_ids[c]] = prev_ids[r]

    used = {seg.speaker for seg in prev_segments} | set(mapping.values())
    for spk in new_speakers:
        if spk in mapping:
            continue
        fresh = _next_free_global_id(used)
        mapping[spk] = fresh
        used.add(fresh)
    return mapping


def _containment(a: Segment, b: Segment) -> float:
    """Temporal intersection over the SHORTER segment's duration.

    Unlike IoU this stays high when one segment is a window-boundary-clipped
    fragment fully inside the other (IoU shrinks with the duration ratio).
    """
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = max(min(a.end - a.start, b.end - b.start), 1e-9)
    return inter / shorter


def _is_duplicate(a: Segment, b: Segment) -> bool:
    """Two segments describe the same speech event (time containment + agreement)."""
    if _containment(a, b) < _DUP_CONTAINMENT:
        return False
    if a.speaker == b.speaker:
        return True
    tok_a = set(tokenize_mixed(a.text))
    tok_b = set(tokenize_mixed(b.text))
    if _jaccard(tok_a, tok_b) >= _DUP_TEXT_JACCARD:
        return True
    # Clipped fragment: the shorter text is (mostly) contained in the longer.
    if tok_a and tok_b:
        shared = len(tok_a & tok_b) / min(len(tok_a), len(tok_b))
        return shared >= _DUP_TEXT_CONTAINMENT
    return False


def _chunk_center(offset: float, segments: list[Segment]) -> float:
    """Approximate temporal center of a chunk from its offset and segment span."""
    end = max((seg.end for seg in segments), default=offset)
    return (offset + end) / 2.0


def _join_texts(a: str, b: str) -> str:
    """Join two texts with a space unless the boundary is CJK-to-CJK."""
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if _CJK_RE.match(a[-1]) and _CJK_RE.match(b[0]):
        return a + b
    return a + " " + b


def _merge_adjacent(segments: list[Segment], gap_s: float = MERGE_GAP_S) -> list[Segment]:
    """Merge consecutive same-speaker segments whose gap is below ``gap_s``."""
    out: list[Segment] = []
    for seg in segments:
        if out and out[-1].speaker == seg.speaker and seg.start - out[-1].end < gap_s:
            last = out[-1]
            out[-1] = replace(
                last, end=max(last.end, seg.end), text=_join_texts(last.text, seg.text)
            )
        else:
            out.append(replace(seg))
    return out


def stitch(
    chunks: list[list[Segment]],
    chunk_offsets: list[float],
    overlap_s: float,
) -> list[Segment]:
    """Stitch per-chunk segment lists into one globally-labeled transcript.

    Expects each chunk's segments in GLOBAL time (already offset by the chunk
    start); ``chunk_offsets`` are the chunk start times, used to locate the
    overlap region ``[chunk_offsets[i], chunk_offsets[i] + overlap_s]``
    between consecutive chunks.

    Per chunk: speakers are remapped via :func:`match_speakers`; duplicated
    segments in the overlap are dropped, keeping the clearly longer version
    (the un-clipped transcription of a boundary-straddling turn) or, for
    near-equal durations, the version from the chunk whose center is closer
    to the segment; finally adjacent same-speaker segments with a gap
    < 0.3 s are merged and the result sorted by time.
    """
    if len(chunks) != len(chunk_offsets):
        raise ValueError(
            f"chunks ({len(chunks)}) and chunk_offsets ({len(chunk_offsets)}) length mismatch"
        )
    if not chunks:
        return []

    # Canonical relabel of the first chunk: SPEAKER_{n} by first appearance.
    first = sorted(chunks[0], key=lambda s: (s.start, s.end))
    relabel: dict[str, str] = {}
    for seg in first:
        relabel.setdefault(seg.speaker, f"SPEAKER_{len(relabel)}")
    merged = [replace(seg, speaker=relabel[seg.speaker]) for seg in first]
    prev_center = _chunk_center(chunk_offsets[0], merged)

    for i in range(1, len(chunks)):
        ov_lo = chunk_offsets[i]
        ov_hi = chunk_offsets[i] + overlap_s
        segs = sorted(chunks[i], key=lambda s: (s.start, s.end))
        mapping = match_speakers(merged, segs, ov_lo, ov_hi)
        segs = [replace(seg, speaker=mapping.get(seg.speaker, seg.speaker)) for seg in segs]
        new_center = _chunk_center(chunk_offsets[i], segs)

        kept: list[Segment] = []
        for seg in segs:
            if seg.end <= ov_lo or seg.start >= ov_hi:
                kept.append(seg)
                continue
            dup_idx = next(
                (
                    j
                    for j, m in enumerate(merged)
                    if m.end > ov_lo and m.start < ov_hi and _is_duplicate(seg, m)
                ),
                None,
            )
            if dup_idx is None:
                kept.append(seg)
            else:
                # Prefer the clearly longer version (the un-clipped
                # transcription of a turn that straddles the window
                # boundary); for near-equal durations fall back to the
                # chunk whose center is closer to the segment.
                dur_new = seg.end - seg.start
                dur_old = merged[dup_idx].end - merged[dup_idx].start
                if dur_new > dur_old * _KEEP_LONGER_RATIO:
                    win_new = True
                elif dur_old > dur_new * _KEEP_LONGER_RATIO:
                    win_new = False
                else:
                    mid = (seg.start + seg.end) / 2.0
                    win_new = abs(mid - new_center) < abs(mid - prev_center)
                if win_new:
                    merged.pop(dup_idx)  # new chunk's version wins
                    kept.append(seg)
                # else: keep the previous chunk's version, drop the new one
        merged.extend(kept)
        prev_center = new_center

    merged.sort(key=lambda s: (s.start, s.end))
    return _merge_adjacent(merged)
