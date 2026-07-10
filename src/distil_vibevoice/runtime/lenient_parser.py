"""Lenient parser for MOSS `[start][Sxx]text[end]` transcript output.

The official ``TranscriptStreamParser`` resets whenever the speaker tag is
missing. On real far-field audio the model sometimes emits ``[start]text[end]``
without a ``[Sxx]`` tag; the strict parser then drops those segments (and often
everything downstream once off-track). This parser accepts both forms and
carries the previous speaker across untagged segments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SEG_RE = re.compile(
    r"\[(?P<start>\d{1,7}(?:\.\d{1,3})?)\]"      # [start]
    r"(?:\[(?P<spk>S\d{1,3})\])?"                # optional [Sxx]
    r"(?P<text>[^\[\]]+?)"                       # text (no brackets inside)
    r"\[(?P<end>\d{1,7}(?:\.\d{1,3})?)\]"        # [end]
)


@dataclass(slots=True)
class LenientSegment:
    start: float
    end: float
    speaker: str
    text: str


def parse_transcript_lenient(text: str) -> list[LenientSegment]:
    segs: list[LenientSegment] = []
    prev_spk = "S01"
    pos = 0
    while True:
        m = _SEG_RE.search(text, pos)
        if not m:
            break
        start, end = float(m.group("start")), float(m.group("end"))
        spk = m.group("spk") or prev_spk
        body = m.group("text").strip()
        if body and end >= start:
            segs.append(LenientSegment(start, end, spk, body))
            prev_spk = spk
        # the closing [end] often doubles as the boundary before the next
        # [start]; resume the scan right AFTER the matched end bracket
        pos = m.end()
    return segs
