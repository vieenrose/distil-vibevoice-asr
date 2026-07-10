"""Tests: summary fidelity metric + grounded target generator (CPU)."""
from __future__ import annotations

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.data.summary_targets import build_notes, build_title, merge_notes
from distil_vibevoice.eval.summary_fidelity import check_fidelity, extract_facts, salient_facts

TRANSCRIPT = (
    "我這邊設備採購的部分大概完成 75%，剩下的預計 週三 之前可以 close。"
    "預算的部分大概要抓 21 萬，4/4 之前要給 quote。"
    "好，我們就這樣定，行銷的 campaign 三月 上線。"
)


def test_extract_facts_finds_pct_amount_date():
    facts = extract_facts(TRANSCRIPT)
    assert "75%" in facts
    assert "21萬" in facts
    assert "4/4" in facts
    assert "週三" in facts
    assert "三月" in facts


def test_faithful_summary_scores_clean():
    summary = "[標題] 預算會議\n[摘要] 設備採購完成 75%，預算 21 萬，4/4 給 quote。"
    r = check_fidelity(TRANSCRIPT, summary)
    assert r.hallucination_rate == 0.0
    assert r.coverage > 0.5
    assert "75%" in r.covered


def test_hallucinated_number_is_caught():
    summary = "[摘要] 設備採購完成 95%，預算 30 萬。"  # 95% and 30萬 not in source
    r = check_fidelity(TRANSCRIPT, summary)
    assert "95%" in r.hallucinated
    assert "30萬" in r.hallucinated
    assert r.hallucination_rate > 0.0


def test_notes_are_faithful_by_construction():
    segs = [
        Segment(0, 5, "0", "今天的 agenda 是設備採購。"),
        Segment(5, 10, "1", "我這邊完成 75%，預算抓 21 萬。"),
        Segment(10, 15, "0", "OK，週五 之前會把 breakdown 寄給大家。"),
        Segment(15, 16, "1", "[Silence]"),
    ]
    notes = build_notes(segs)
    transcript = " ".join(s.text for s in segs)
    r = check_fidelity(transcript, notes)
    assert r.hallucination_rate == 0.0  # notes only quote the source
    assert "[主題]" in notes and "[重點]" in notes and "[待辦]" in notes


def test_merge_dedups_and_structures():
    n1 = "[主題] agenda 是 A\n[重點] 0: 完成 75%\n[待辦] 1: 週五 之前寄出"
    n2 = "[重點] 0: 完成 75%\n[待辦] 1: 追蹤 21 萬 的預算"  # dup keypoint
    merged = merge_notes([n1, n2], build_title("財務檢討會議", "月結的應收帳款"))
    assert merged.count("完成 75%") == 1  # deduped
    assert merged.startswith("[標題] 財務檢討會議：月結的應收帳款")
    assert "待辦事項：" in merged
    # merged summary is faithful to the union of its notes
    r = check_fidelity(n1 + "\n" + n2, merged)
    assert r.hallucination_rate == 0.0


def test_timestamp_markup_not_counted_as_facts():
    from distil_vibevoice.eval.summary_fidelity import extract_facts
    # transcript-style output: timestamps must not become "facts"
    ts_text = "[0.00][S01]設備採購完成 75%[10.17][11.58][S02]預算 21 萬[22.98]"
    facts = extract_facts(ts_text)
    assert "0.00" not in facts and "10.17" not in facts and "22.98" not in facts
    assert "75%" in facts and "21萬" in facts  # real facts survive


def test_transcript_style_output_is_faithful_to_itself():
    from distil_vibevoice.eval.summary_fidelity import check_fidelity
    transcript = "設備採購完成 75%，預算 21 萬。"
    # model (wrongly) emits transcript style, but content matches -> not hallucinated
    hyp = "[0.00][S01]設備採購完成 75%[5.0][5.2][S02]預算 21 萬[10.0]"
    r = check_fidelity(transcript, hyp)
    assert r.hallucination_rate == 0.0
