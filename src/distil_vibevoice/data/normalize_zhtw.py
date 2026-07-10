"""Normalize Chinese text to Traditional/Taiwan (zh-TW) via OpenCC ``s2twp``.

English/ASCII spans are protected from conversion: text is split into runs of
``[A-Za-z0-9 .,'-]+`` (passed through untouched) and everything else (fed to
the converter). The OpenCC converter is lazily initialized once per process.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from distil_vibevoice.data.manifest import MeetingRecord, Segment

__all__ = ["to_zhtw", "normalize_record"]

# Runs protected from OpenCC conversion (English words, digits, basic ASCII
# punctuation commonly interleaved in code-switched text).
_PROTECTED_RE = re.compile(r"[A-Za-z0-9 .,'\-]+")

# Supplemental Taiwan-phrase overrides applied after OpenCC s2twp. Upstream
# OpenCC dropped some ambiguous mappings (e.g. 質量 also means "mass" in
# physics), but in meeting-transcript context the Taiwan business phrasing is
# what we want. Applied on the converted (Traditional) text.
_TW_PHRASE_OVERRIDES: dict[str, str] = {
    "質量": "品質",  # quality (mainland 质量) -> Taiwan 品質
}

_converter: Any = None


def _get_converter() -> Any:
    """Lazily create and cache the module-level OpenCC s2twp converter."""
    global _converter
    if _converter is None:
        try:
            import opencc
        except ImportError as e:  # pragma: no cover - exercised only without opencc
            raise ImportError(
                "OpenCC is required for zh-TW normalization. "
                "Install it with: pip install opencc  (or: pip install opencc-python-reimplemented)"
            ) from e
        try:
            _converter = opencc.OpenCC("s2twp")
        except Exception:
            # opencc-python-reimplemented expects the '.json' suffix.
            _converter = opencc.OpenCC("s2twp.json")
    return _converter


def to_zhtw(text: str) -> str:
    """Convert text to Traditional Chinese (Taiwan phrasing, OpenCC s2twp).

    ASCII/English runs matching ``[A-Za-z0-9 .,'-]+`` are passed through
    verbatim so code-switched English is never mangled by the converter.
    """
    if not text:
        return text
    conv = _get_converter()
    parts: list[str] = []
    pos = 0
    for m in _PROTECTED_RE.finditer(text):
        if m.start() > pos:
            parts.append(conv.convert(text[pos : m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        parts.append(conv.convert(text[pos:]))
    out = "".join(parts)
    for src, dst in _TW_PHRASE_OVERRIDES.items():
        out = out.replace(src, dst)
    return out


def normalize_record(rec: MeetingRecord) -> MeetingRecord:
    """Return a copy of ``rec`` with every segment text normalized to zh-TW.

    The input record is not mutated.
    """
    segments = [
        Segment(start=s.start, end=s.end, speaker=s.speaker, text=to_zhtw(s.text))
        for s in rec.segments
    ]
    return replace(rec, segments=segments, meta=dict(rec.meta))
