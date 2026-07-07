"""Tests for release gates (run_gates / GateReport) and timestamp MAE."""

from __future__ import annotations

from pathlib import Path

from distil_vibevoice.data.manifest import MeetingRecord, Segment, write_manifest
from distil_vibevoice.eval.gates import GateReport, run_gates
from distil_vibevoice.eval.timestamps import timestamp_mae, timestamp_report


def fixture_meeting() -> list[Segment]:
    """Tiny 3-speaker zh+en code-switched meeting used across eval tests."""
    return [
        Segment(start=0.0, end=3.0, speaker="A", text="大家好 歡迎參加今天的會議"),
        Segment(start=3.0, end=6.5, speaker="B", text="we should review the roadmap first"),
        Segment(start=6.5, end=10.0, speaker="C", text="好的 我來報告 progress"),
        Segment(start=10.0, end=13.0, speaker="A", text="thanks 請開始"),
        Segment(start=13.0, end=16.0, speaker="B", text="第一個 milestone 已經完成"),
    ]


def make_record(
    audio_path: str,
    segments: list[Segment],
    duration_s: float = 16.0,
    meta: dict | None = None,
) -> MeetingRecord:
    return MeetingRecord(
        audio_path=audio_path,
        duration_s=duration_s,
        sample_rate=24000,
        language="zh-TW-en",
        source="unit-test",
        split="test",
        segments=segments,
        meta=meta or {},
    )


def write_pair(
    tmp_path: Path,
    ref_records: list[MeetingRecord],
    hyp_records: list[MeetingRecord],
) -> tuple[str, str]:
    ref_path = tmp_path / "ref.jsonl"
    hyp_path = tmp_path / "hyp.jsonl"
    write_manifest(ref_records, ref_path)
    write_manifest(hyp_records, hyp_path)
    return str(ref_path), str(hyp_path)


# ---------------------------------------------------------------------------
# timestamp_mae / timestamp_report
# ---------------------------------------------------------------------------


def test_timestamp_mae_perfect_is_zero() -> None:
    ref = fixture_meeting()
    assert timestamp_mae(ref, fixture_meeting()) == 0.0


def test_timestamp_mae_uniform_shift() -> None:
    ref = fixture_meeting()
    hyp = [Segment(s.start + 0.1, s.end + 0.1, s.speaker, s.text) for s in ref]
    assert abs(timestamp_mae(ref, hyp) - 0.1) < 1e-9


def test_timestamp_report_counts_unmatched() -> None:
    ref = fixture_meeting()
    hyp = fixture_meeting()[:3]  # last two ref segments have no hyp match
    report = timestamp_report(ref, hyp)
    assert report["n_matched"] == 3
    assert report["n_ref_unmatched"] == 2
    assert report["n_hyp_unmatched"] == 0
    assert report["mae"] == 0.0


def test_timestamp_dissimilar_text_not_matched() -> None:
    ref = [Segment(0.0, 2.0, "A", "完全不同的內容在這裡")]
    hyp = [Segment(0.0, 2.0, "A", "totally different words")]
    report = timestamp_report(ref, hyp)
    assert report["n_matched"] == 0
    assert timestamp_mae(ref, hyp) == 0.0


def test_timestamp_matching_survives_speaker_relabel() -> None:
    ref = fixture_meeting()
    hyp = [
        Segment(s.start + 0.2, s.end + 0.2, {"A": "0", "B": "1", "C": "2"}[s.speaker], s.text)
        for s in ref
    ]
    assert abs(timestamp_mae(ref, hyp) - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# run_gates
# ---------------------------------------------------------------------------


def test_gates_pass_on_perfect_hyp(tmp_path: Path) -> None:
    ref = [make_record("m0.wav", fixture_meeting())]
    hyp = [make_record("m0.wav", fixture_meeting())]
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    thresholds = {"mer": 0.1, "cpwer": 0.1, "der": 0.1, "timestamp_mae": 0.5}
    report = run_gates(ref_path, hyp_path, thresholds)
    assert isinstance(report, GateReport)
    assert report.passed
    assert report.failures == []
    assert report.metrics["mer"] == 0.0
    assert report.metrics["cpwer"] == 0.0
    assert report.metrics["der"] == 0.0
    assert report.metrics["timestamp_mae"] == 0.0
    assert report.metrics["n_records"] == 1


def test_gates_fail_lists_violated_metrics(tmp_path: Path) -> None:
    ref = [make_record("m0.wav", fixture_meeting())]
    # Hypothesis misses one speaker entirely and garbles a segment.
    bad_segments = [s for s in fixture_meeting() if s.speaker != "B"]
    bad_segments[0] = Segment(0.0, 3.0, "A", "完全錯誤的辨識結果啦")
    hyp = [make_record("m0.wav", bad_segments)]
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    thresholds = {"mer": 0.05, "cpwer": 0.05, "der": 0.05}
    report = run_gates(ref_path, hyp_path, thresholds)
    assert not report.passed
    failed_metrics = {f.split()[0] for f in report.failures}
    assert failed_metrics == {"mer", "cpwer", "der"}
    for failure in report.failures:
        name, value, gt, limit = failure.split()
        assert gt == ">"
        assert float(value) > float(limit)


def test_gates_missing_hyp_record_scored_worst_case(tmp_path: Path) -> None:
    ref = [
        make_record("m0.wav", fixture_meeting()),
        make_record("m1.wav", fixture_meeting()),
    ]
    hyp = [make_record("m0.wav", fixture_meeting())]  # m1 missing
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    report = run_gates(ref_path, hyp_path, {"mer": 0.1})
    assert not report.passed
    assert report.metrics["n_missing_hyp"] == 1
    # Equal durations: mean of 0.0 and 1.0.
    assert abs(report.metrics["mer"] - 0.5) < 1e-9


def test_gates_duration_weighted_aggregation(tmp_path: Path) -> None:
    perfect = fixture_meeting()
    ref = [
        make_record("long.wav", perfect, duration_s=90.0),
        make_record("short.wav", perfect, duration_s=10.0),
    ]
    hyp = [
        make_record("long.wav", perfect, duration_s=90.0),
        make_record("short.wav", [], duration_s=10.0),  # mer 1.0 on the short one
    ]
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    report = run_gates(ref_path, hyp_path, {"mer": 0.5})
    # Weighted: (0.0*90 + 1.0*10) / 100 = 0.1 -> passes a 0.5 gate.
    assert abs(report.metrics["mer"] - 0.1) < 1e-9
    assert report.passed


def test_gates_codeswitch_slice(tmp_path: Path) -> None:
    perfect = fixture_meeting()
    garbled = [Segment(s.start, s.end, s.speaker, "亂七八糟 wrong text") for s in perfect]
    ref = [
        make_record("plain.wav", perfect),
        make_record("cs.wav", perfect, meta={"slice": "codeswitch"}),
    ]
    hyp = [
        make_record("plain.wav", perfect),
        make_record("cs.wav", garbled),
    ]
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    # Overall mer passes (only half the duration is bad) but the codeswitch
    # slice alone fails its dedicated gate.
    thresholds = {"mer": 0.6, "codeswitch_mer": 0.1}
    report = run_gates(ref_path, hyp_path, thresholds)
    assert not report.passed
    assert "codeswitch_mer" in report.metrics
    assert report.metrics["codeswitch_mer"] > 0.1
    assert len(report.failures) == 1
    assert report.failures[0].startswith("codeswitch_mer ")


def test_gates_codeswitch_threshold_skipped_when_slice_absent(tmp_path: Path) -> None:
    ref = [make_record("m0.wav", fixture_meeting())]
    hyp = [make_record("m0.wav", fixture_meeting())]
    ref_path, hyp_path = write_pair(tmp_path, ref, hyp)
    report = run_gates(ref_path, hyp_path, {"mer": 0.1, "codeswitch_mer": 0.1})
    assert report.passed
    assert "codeswitch_mer" not in report.metrics


def test_gates_empty_ref_manifest_fails(tmp_path: Path) -> None:
    ref_path, hyp_path = write_pair(tmp_path, [], [])
    report = run_gates(ref_path, hyp_path, {"mer": 0.1})
    assert not report.passed
    assert report.failures
