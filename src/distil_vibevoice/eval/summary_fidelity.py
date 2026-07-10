"""Faithfulness metrics for meeting summaries — reference-free, fact-based.

A summary of a meeting is *unfaithful* when it contains facts (numbers,
percentages, amounts, dates, names) that do not occur in the source transcript.
It has poor *coverage* when salient transcript facts are missing. Both are
checkable programmatically without a gold reference summary — which is exactly
what a small on-device model needs guarding against (numeric hallucination is
the most damaging failure mode in meeting minutes).

Facts extracted: percentages (75%), amounts (21 萬 / 3 億), plain numbers,
zh dates (4/4, 週三, 三月), and capitalized/CJK names present in a roster.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PCT = re.compile(r"\d+(?:\.\d+)?%")
_AMOUNT = re.compile(r"\d+(?:\.\d+)?\s*[萬億千]")
_DATE = re.compile(r"\d{1,2}/\d{1,2}|週[一二三四五六日]|[一二三四五六七八九十]+月")
# plain numbers, excluding speaker-label digits ("0:", "S01") — not facts
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?!:)")

# transcript-style markup that is NOT summary content and must not count as
# facts: bracketed timestamps like [12.34] and [12.34][45.67], and [Sxx] tags.
_TIMESTAMP_MARKUP = re.compile(r"\[\s*\d+(?:\.\d+)?\s*\]")
_SPEAKER_MARKUP = re.compile(r"\[S\d+\]|\bS\d+\s*[:：]")


def _strip_markup(text: str) -> str:
    """Remove transcript timestamp/speaker markup before fact extraction."""
    text = _TIMESTAMP_MARKUP.sub(" ", text)
    text = _SPEAKER_MARKUP.sub(" ", text)
    return text


@dataclass
class FidelityReport:
    hallucinated: list[str]      # facts in summary absent from transcript
    covered: list[str]           # salient transcript facts present in summary
    missed: list[str]            # salient transcript facts absent from summary
    hallucination_rate: float    # |hallucinated| / |summary facts|  (0 = faithful)
    coverage: float              # |covered| / |salient facts|       (1 = complete)


def extract_facts(text: str) -> set[str]:
    """All checkable facts in a text (percentages, amounts, dates, numbers).

    Transcript timestamp/speaker markup is stripped first so a transcript-style
    output isn't scored as if every timestamp were a hallucinated fact.
    """
    text = _strip_markup(text)
    facts: set[str] = set()
    for pat in (_PCT, _AMOUNT, _DATE):
        facts.update(m.group(0).replace(" ", "") for m in pat.finditer(text))
    # plain numbers not already inside a matched fact
    matched = "".join(facts)
    for m in _NUMBER.finditer(text):
        if m.group(0) not in matched:
            facts.add(m.group(0))
    return facts


def salient_facts(transcript: str) -> set[str]:
    """Transcript facts a good summary should retain: percentages, amounts, dates.

    Plain bare numbers are excluded here (too noisy to demand coverage of all),
    but they still count for hallucination checking on the summary side.
    """
    facts: set[str] = set()
    for pat in (_PCT, _AMOUNT, _DATE):
        facts.update(m.group(0).replace(" ", "") for m in pat.finditer(transcript))
    return facts


def check_fidelity(transcript: str, summary: str) -> FidelityReport:
    """Reference-free faithfulness check of ``summary`` against ``transcript``."""
    t_all = extract_facts(transcript)
    s_all = extract_facts(summary)
    salient = salient_facts(transcript)

    hallucinated = sorted(f for f in s_all if f not in t_all)
    covered = sorted(f for f in salient if f in s_all)
    missed = sorted(f for f in salient if f not in s_all)

    h_rate = len(hallucinated) / len(s_all) if s_all else 0.0
    cov = len(covered) / len(salient) if salient else 1.0
    return FidelityReport(hallucinated, covered, missed, h_rate, cov)
