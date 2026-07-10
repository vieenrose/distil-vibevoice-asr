"""Vocabulary-coverage analysis for the deferred student vocab-trim decision.

Given real training targets (rendered with :func:`format_target`) and a
tokenizer, count which teacher-vocab ids are actually emitted and decide which
ids a trimmed-vocab student should keep. This informs whether hard
vocab-trimming (dropping unused embedding/lm-head rows) is safe: if coverage of
the emitted token-occurrences by the kept set is ~1.0, the dropped ids never
appear in targets and trimming loses nothing at train time.

The kept set is always the UNION of

* ids whose occurrence count is ``>= min_count``,
* every byte-fallback token (the 256 ``<0xNN>`` pieces) when ``include_bytes``
  — these guarantee any UTF-8 byte stays representable even if unseen,
* every special / added token id,
* any explicit ``extra_ids``.

The tokenizer is **passed in** (never downloaded here); the optional
``transformers`` import is deferred and only used by the convenience helpers
that build one on request.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .manifest import format_target, iter_manifest

__all__ = ["build_kept_vocab", "analyze_manifest_vocab"]

# Byte-fallback tokens look like ``<0x0A>`` (SentencePiece byte pieces).
_BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


# ---------------------------------------------------------------------------
# Tokenizer introspection helpers (all defensive: unknown tokenizers degrade
# gracefully rather than raising).
# ---------------------------------------------------------------------------
def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` to a flat list of ids without special tokens."""
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        try:
            return list(enc(text, add_special_tokens=False))
        except TypeError:
            return list(enc(text))
    out = tokenizer(text)
    ids = out["input_ids"] if isinstance(out, dict) else out
    return list(ids)


def _get_vocab(tokenizer: Any) -> dict[str, int]:
    getter = getattr(tokenizer, "get_vocab", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:  # pragma: no cover - defensive
            return {}
    vocab = getattr(tokenizer, "vocab", None)
    return dict(vocab) if isinstance(vocab, dict) else {}


def _byte_token_ids(tokenizer: Any) -> set[int]:
    return {
        int(tid)
        for tok, tid in _get_vocab(tokenizer).items()
        if isinstance(tok, str) and _BYTE_TOKEN_RE.match(tok)
    }


def _special_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    special = getattr(tokenizer, "all_special_ids", None)
    if special:
        ids.update(int(x) for x in special)
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        try:
            ids.update(int(v) for v in added().values())
        except Exception:  # pragma: no cover - defensive
            pass
    return ids


def _vocab_size(tokenizer: Any) -> int:
    size = getattr(tokenizer, "vocab_size", None)
    candidates = [size if isinstance(size, int) else 0]
    try:
        candidates.append(len(tokenizer))
    except TypeError:
        pass
    vocab = _get_vocab(tokenizer)
    if vocab:
        candidates.append(max(vocab.values()) + 1)
    return max(candidates)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def build_kept_vocab(
    manifest_paths: list[str],
    tokenizer: Any,
    min_count: int = 1,
    include_bytes: bool = True,
    extra_ids: "set[int] | None" = None,
) -> dict:
    """Decide which teacher-vocab ids a trimmed student should keep.

    Tokenizes ``format_target(record.segments)`` for every record across all
    ``manifest_paths`` and counts token ids. Returns a dict with:

    * ``kept_ids``: sorted list of kept ids (frequent ids UNION byte-fallback
      UNION special/added UNION ``extra_ids``),
    * ``coverage``: fraction of total emitted token-occurrences whose id is in
      the kept set (``1.0`` at ``min_count=1`` — nothing frequent is dropped),
    * ``dropped_high_freq``: ``(id, count)`` pairs for *emitted* ids that were
      dropped, sorted by descending count (empty at ``min_count=1``); flags ids
      you are about to trim away that nonetheless appear in targets,
    * ``n_kept``: ``len(kept_ids)``,
    * ``n_total``: tokenizer vocabulary size,
    * ``token_counts``: ``{id: count}`` over all emitted occurrences.
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    counts: Counter[int] = Counter()
    for path in manifest_paths:
        for record in iter_manifest(path):
            target = format_target(record.segments)
            counts.update(_encode(tokenizer, target))

    total_occurrences = sum(counts.values())

    always_keep: set[int] = set()
    if include_bytes:
        always_keep |= _byte_token_ids(tokenizer)
    always_keep |= _special_token_ids(tokenizer)
    if extra_ids:
        always_keep |= {int(x) for x in extra_ids}

    frequent = {tid for tid, c in counts.items() if c >= min_count}
    kept = frequent | always_keep

    # Emitted ids that fell below threshold and are not force-kept.
    dropped_high_freq = sorted(
        ((tid, c) for tid, c in counts.items() if tid not in kept),
        key=lambda item: (-item[1], item[0]),
    )

    kept_occurrences = sum(c for tid, c in counts.items() if tid in kept)
    coverage = 1.0 if total_occurrences == 0 else kept_occurrences / total_occurrences

    n_total = max(_vocab_size(tokenizer), (max(kept) + 1) if kept else 0)

    return {
        "kept_ids": sorted(kept),
        "coverage": coverage,
        "dropped_high_freq": dropped_high_freq,
        "n_kept": len(kept),
        "n_total": n_total,
        "token_counts": dict(counts),
    }


def analyze_manifest_vocab(
    manifest_paths: list[str],
    tokenizer: Any,
    min_count: int = 1,
    include_bytes: bool = True,
    extra_ids: "set[int] | None" = None,
    hidden_size: int = 1536,
    dtype_bytes: int = 2,
    tied_embeddings: bool = True,
) -> dict:
    """:func:`build_kept_vocab` plus projected param / RAM savings from trimming.

    Savings assume the student keeps ``n_kept`` of ``n_total`` rows in each
    ``hidden_size``-wide embedding matrix. With tied embeddings there is a
    single shared matrix; untied counts both ``embed_tokens`` and ``lm_head``.
    ``dtype_bytes`` (default 2 for bf16/fp16) converts the row saving to bytes.
    """
    result = build_kept_vocab(
        manifest_paths,
        tokenizer,
        min_count=min_count,
        include_bytes=include_bytes,
        extra_ids=extra_ids,
    )
    matrices = 1 if tied_embeddings else 2
    rows_dropped = result["n_total"] - result["n_kept"]
    params_saved = rows_dropped * hidden_size * matrices
    ram_saved_bytes = params_saved * dtype_bytes
    result["projected_savings"] = {
        "hidden_size": hidden_size,
        "tied_embeddings": tied_embeddings,
        "rows_dropped": rows_dropped,
        "params_saved": params_saved,
        "ram_saved_bytes": ram_saved_bytes,
        "ram_saved_mb": ram_saved_bytes / (1024 * 1024),
    }
    return result
