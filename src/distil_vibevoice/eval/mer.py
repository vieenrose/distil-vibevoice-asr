"""Mixed error rate (MER) for zh/en code-switched transcripts.

Tokenization treats every CJK character as one token and every latin word /
digit run as one token, so Chinese errors are counted per character and
English errors per word.  Punctuation (ASCII and fullwidth) and whitespace
are separators and never counted.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import numpy as np

__all__ = ["tokenize_mixed", "levenshtein", "mer"]

# One CJK ideograph per token (URO + Ext-A + 〇 + compatibility ideographs +
# Ext-B..Ext-H), latin words (incl. apostrophes), or digit runs.  Everything
# else (spaces, ASCII/fullwidth punctuation such as `，。！？；：「」（）、`,
# × U+00D7, ÷ U+00F7 ...) is a separator because it simply never matches.
# NFKC normalization (applied in tokenize_mixed) folds fullwidth latin letters,
# fullwidth digits and compatibility punctuation to their ASCII forms first.
_TOKEN_RE = re.compile(
    "[一-鿿㐀-䶿〇\uf900-\ufaff\U00020000-\U0003134f]"  # CJK: one char = one token
    "|[a-zà-öø-ɏ]+(?:['’][a-zà-öø-ɏ]+)*"  # latin word (× / ÷ excluded)
    "|[0-9]+"  # digit run
)


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize mixed zh/en text: CJK chars, latin words, digit runs.

    Lowercases, NFKC-normalizes (fullwidth -> halfwidth) and strips all
    punctuation, so tokenization is spacing- and punctuation-insensitive
    for Chinese.
    """
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Token-level edit distance, O(len(ref)*len(hyp)) with numpy row rolling."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    # Map tokens to small ints so the inner comparison is vectorized.
    vocab: dict[str, int] = {}
    a = np.fromiter(
        (vocab.setdefault(t, len(vocab)) for t in ref), dtype=np.int64, count=len(ref)
    )
    b = np.fromiter(
        (vocab.setdefault(t, len(vocab)) for t in hyp), dtype=np.int64, count=len(hyp)
    )
    n = b.size
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    cur = np.empty(n + 1, dtype=np.int64)
    for token in a:
        cost = (b != token).astype(np.int64)
        cur[0] = prev[0] + 1
        # Deletion (prev[1:]+1) and substitution/match (prev[:-1]+cost).
        np.minimum(prev[1:] + 1, prev[:-1] + cost, out=cur[1:])
        # Insertions: enforce cur[j] = min_{k<=j}(cur[k] + (j-k)) via the
        # classic "subtract index, prefix-min, add index" scan.
        cur = np.minimum.accumulate(cur - idx) + idx
        prev, cur = cur, prev
    return int(prev[-1])


def mer(ref: str, hyp: str) -> float:
    """Mixed error rate: edit distance over mixed tokens / #ref tokens.

    Chinese is scored per character, English per word (see tokenize_mixed).
    Edge cases: empty ref and empty hyp -> 0.0; empty ref with non-empty
    hyp -> 1.0.
    """
    ref_tokens = tokenize_mixed(ref)
    hyp_tokens = tokenize_mixed(hyp)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)
