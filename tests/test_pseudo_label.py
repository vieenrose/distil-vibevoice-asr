"""Tests for the local mixed tokenizer / two-pass agreement in data.pseudo_label.

Only the pure-python filtering helpers are exercised (TeacherLabeler needs the
vibevoice package and a GPU and is not imported at module level).
"""
from __future__ import annotations

from distil_vibevoice.data.manifest import MeetingRecord, Segment
from distil_vibevoice.data.pseudo_label import _tokenize_mixed, two_pass_agreement


def _rec(*texts: str) -> MeetingRecord:
    segments = [
        Segment(start=float(i), end=float(i) + 1.0, speaker="S0", text=t)
        for i, t in enumerate(texts)
    ]
    return MeetingRecord(
        audio_path="a.wav",
        duration_s=float(len(texts)),
        sample_rate=16000,
        language="zh-TW-en",
        source="test",
        split="train",
        segments=segments,
        meta={},
    )


def test_tokenize_folds_fullwidth_latin_and_digits() -> None:
    # Fullwidth forms (common in CJK teacher transcripts) must be tokenized,
    # not silently dropped.
    assert _tokenize_mixed("ＫＰＩ達標了") == ["kpi", "達", "標", "了"]
    assert _tokenize_mixed("成長１２０％") == ["成", "長", "120"]


def test_tokenize_matches_canonical_eval_tokenizer() -> None:
    from distil_vibevoice.eval.mer import tokenize_mixed as eval_tokenize

    samples = [
        "這一季的ＫＰＩ達標了",
        "成長１２０％",
        "二〇二五年 Q3 的 roadmap",
        "don't stop reviewing 3×4",
        "Hello, World! 你好。",
    ]
    for text in samples:
        assert _tokenize_mixed(text) == eval_tokenize(text), text


def test_two_pass_agreement_sees_fullwidth_disagreement() -> None:
    # The two passes fully disagree on the (fullwidth) English term; the
    # filter must reject the record instead of scoring an edit distance of 0.
    a = _rec("這一季的ＫＰＩ達標了")
    b = _rec("這一季的ＧＤＰ達標了")
    assert not two_pass_agreement(a, b, max_cpwer=0.05)


def test_two_pass_agreement_is_width_insensitive() -> None:
    # Fullwidth vs halfwidth renderings of the same transcript agree exactly.
    a = _rec("這一季的ＫＰＩ達標了")
    b = _rec("這一季的 KPI 達標了")
    assert two_pass_agreement(a, b, max_cpwer=0.0)


def test_two_pass_agreement_basic() -> None:
    assert two_pass_agreement(_rec("大家好"), _rec("大家好"), max_cpwer=0.0)
    assert not two_pass_agreement(_rec("大家好"), _rec("完全不同"), max_cpwer=0.05)
    assert two_pass_agreement(_rec(), _rec(), max_cpwer=0.0)  # both empty
    assert not two_pass_agreement(_rec("有話"), _rec(), max_cpwer=0.5)
