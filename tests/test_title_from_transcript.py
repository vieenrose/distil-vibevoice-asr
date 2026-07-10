"""Tests for deterministic title extraction (CPU, no model)."""
from distil_vibevoice.data.title_from_transcript import title_from_transcript


def test_type_and_topic_from_opening():
    t = "大家好，我們開始今天的產品站立會議，先請每個人快速 update 一下進度。後面很多其它內容。"
    assert title_from_transcript(t) == "產品站立會議"


def test_topic_from_agenda_cue():
    t = "各位早，這次 meeting 主要是 review CI 的 pipeline，時間控制在三十分鐘內。之後略。"
    title = title_from_transcript(t)
    assert "CI 的 pipeline" in title


def test_strips_determiner_prefix():
    t = "不好意思讓大家久等，今天的預算規劃會議直接開始。"
    # must not keep 今天的
    assert title_from_transcript(t).startswith("預算規劃會議")


def test_fallback_when_atypical():
    assert title_from_transcript("嗯，那個，就這樣吧。") == "會議"
