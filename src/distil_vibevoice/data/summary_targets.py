"""Grounded summary/title targets for meeting-summarization fine-tuning.

Long-meeting design (map-reduce, hallucination-resistant):
  MAP    per window -> structured NOTES (topics / decisions / action items /
         key numbers) that QUOTE facts verbatim from the transcript, and
  REDUCE merged notes -> title + final summary (text-only pass, any length).

Because our TTS corpus is template-generated, the *roles* of turns are
recoverable from their surface forms, so notes/summaries are derivable
deterministically — gold targets without human annotation or an LLM. Facts
(percentages, amounts, dates) are copied verbatim, which is what teaches a
small model extractive fidelity instead of free composition.
"""
from __future__ import annotations

import re

from distil_vibevoice.eval.summary_fidelity import salient_facts

_TITLE_TMPL = "{domain_zh}：{topic}"

# Surface markers of turn roles in the template grammar (and, approximately,
# in real meetings too — these are generic meeting-speech cues).
_DECISION_CUES = ("決議", "就這樣定", "定案", "拍板", "我們就", "決定")
_ACTION_CUES = ("之前", "會把", "負責", "追蹤", "跟進", "follow up", "交給", "寄給")
_TOPIC_CUES = ("agenda", "主要是", "議題", "討論")

_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def build_notes(segments: list) -> str:
    """Structured, grounded notes for ONE window of transcript segments.

    segments: objects/dicts with .speaker/.text (Segment or manifest dicts).
    Output format (the MAP target the model learns):

        [主題] ...
        [重點] speaker: fact-bearing line
        [待辦] speaker: action line
    """
    def spk(s):
        return s["speaker"] if isinstance(s, dict) else s.speaker

    def txt(s):
        return s["text"] if isinstance(s, dict) else s.text

    # Deduplicate by structural signature (drop speaker prefix, digits, and
    # non-CJK/latin punctuation): templated dialogue reuses sentence skeletons,
    # and repetitive targets teach a small model to loop under greedy decoding.
    seen_sig: set[str] = set()

    def _fresh(line: str) -> bool:
        sig = re.sub(r"[\d\s]", "", re.sub(r"^[^:：]*[:：]", "", line))[:24]
        if sig in seen_sig:
            return False
        seen_sig.add(sig)
        return True

    topics: list[str] = []
    keypoints: list[str] = []
    actions: list[str] = []
    for s in segments:
        t = _clean(txt(s))
        if not t or t.startswith("["):
            continue
        low = t.lower()
        if any(c in t for c in _TOPIC_CUES) and len(topics) < 3 and _fresh(t):
            topics.append(t)
        elif any(c in t or c in low for c in _ACTION_CUES) and _fresh(t):
            actions.append(f"{spk(s)}: {t}")
        elif salient_facts(t) and _fresh(t):  # fact line -> keypoint, verbatim
            keypoints.append(f"{spk(s)}: {t}")

    lines: list[str] = []
    for t in topics[:2]:
        lines.append(f"[主題] {t}")
    for k in keypoints[:6]:
        lines.append(f"[重點] {k}")
    for a in actions[:6]:
        lines.append(f"[待辦] {a}")
    if not lines:
        lines.append("[主題] （本段無重點事項）")
    return "\n".join(lines)


def build_title(domain_zh: str, topic: str) -> str:
    return _TITLE_TMPL.format(domain_zh=domain_zh, topic=topic)


def merge_notes(window_notes: list[str], title: str) -> str:
    """The REDUCE target: merged notes -> final title + summary.

    Deduplicates repeated lines across windows, keeps order of first
    appearance, groups by section.
    """
    seen: set[str] = set()
    topics: list[str] = []
    keypoints: list[str] = []
    actions: list[str] = []
    for notes in window_notes:
        for line in notes.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            if line.startswith("[主題]"):
                topics.append(line[4:].strip())
            elif line.startswith("[重點]"):
                keypoints.append(line[4:].strip())
            elif line.startswith("[待辦]"):
                actions.append(line[4:].strip())

    out = [f"[標題] {title}", "[摘要]"]
    if topics:
        out.append("議題：" + "；".join(topics[:3]))
    for k in keypoints[:8]:
        out.append("• " + k)
    if actions:
        out.append("待辦事項：")
        for a in actions[:8]:
            out.append("- " + a)
    return "\n".join(out)
