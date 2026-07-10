"""Deterministic meeting title from a transcript — no model needed.

A 0.6B audio model can transcribe faithfully but cannot reliably *reformat* the
meeting into a title (it degenerates under the transcription prior). But the
transcript already names the meeting: the opening turn states the type
("今天的產品站立會議…") and the agenda/topic. So we extract the title from the
transcript text with a small rule set — far more reliable than asking the model.

title_from_transcript(text) -> "會議類型：主題"  (best effort; type or topic may
be empty if the opening is atypical).
"""
from __future__ import annotations

import re

# Meeting-type words. Use the shortest 2-4 char run immediately before 會議 so
# "今天的產品站立會議" -> "產品站立會議" (not "天的產品站立會議").
_TYPE_RE = re.compile(r"([一-鿿]{2,5}?會議)")

# Agenda/topic cue phrases; the span after them is the topic. Ordered by how
# reliably the phrase marks the *meeting's* topic (vs a per-turn mention).
_TOPIC_CUES = [
    r"agenda\s*是\s*([^，。？\n]+?)(?:。|，|$)",
    r"主要要\s*align\s*([^，。？\n]+?)(?:，|還|。)",
    r"主要是\s*review\s*([^，。？\n]+?)(?:，|。)",
    r"今天主要要\s*([^，。？\n]+?)(?:，|。)",
    r"議題是\s*([^，。？\n]+?)(?:。|，|$)",
]


def _clean_topic(t: str) -> str:
    t = t.strip()
    # drop trailing time-box boilerplate and English filler tails
    t = re.split(r"時間控制|，還|還有確認|，先請", t)[0].strip()
    return t[:24]


def title_from_transcript(text: str, opening_chars: int = 120) -> str:
    """Best-effort 'type: topic' title from transcript text.

    The meeting type and agenda are stated in the OPENING; searching only the
    first ``opening_chars`` avoids picking up per-turn topic mentions later in
    the meeting.
    """
    head = text[:opening_chars]
    # search the whole transcript for the type word (some openings use an
    # agenda cue instead of naming the type, but a later turn names it).
    type_m = _TYPE_RE.search(head) or _TYPE_RE.search(text)
    mtype = type_m.group(1) if type_m else ""
    # strip a leading determiner the greedy class may include (的/今天的/這次…)
    mtype = re.sub(r"^(?:今天的|這次的?|本次的?|的)", "", mtype)

    topic = ""
    for pat in _TOPIC_CUES:
        m = re.search(pat, head, flags=re.IGNORECASE)
        if m:
            topic = _clean_topic(m.group(1))
            if topic:
                break

    if mtype and topic:
        return f"{mtype}：{topic}"
    if mtype:
        return mtype
    if topic:
        return f"會議：{topic}"
    return "會議"
