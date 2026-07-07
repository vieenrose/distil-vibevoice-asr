"""Tests for distil_vibevoice.data.normalize_zhtw (OpenCC s2twp + protection)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest.importorskip("opencc")

from distil_vibevoice.data.manifest import MeetingRecord, Segment
from distil_vibevoice.data.normalize_zhtw import normalize_record, to_zhtw


class TestToZhtw:
    def test_simplified_to_taiwan_phrases(self):
        out = to_zhtw("软件视频质量")
        assert "軟體" in out  # 软件 -> 軟體 (Taiwan phrasing, not 軟件)
        assert "品質" in out  # 质量 -> 品質 (not 質量)
        assert "软件" not in out
        assert "质量" not in out
        # 视频 -> Taiwan phrasing (視訊/影片 depending on OpenCC dict version).
        assert "视频" not in out and ("視訊" in out or "影片" in out)

    def test_char_level_conversion(self):
        assert to_zhtw("会议记录") == "會議記錄"

    def test_english_protected_inside_zh(self):
        text = "我们需要align timeline在下周之前"
        out = to_zhtw(text)
        assert "align timeline" in out
        assert "我們" in out
        assert "下週" in out or "下周" in out

    def test_protected_run_verbatim(self):
        # A pure-ASCII protected run must pass through completely untouched.
        text = "请check一下GPT-4, o3.5 models' output哦"
        out = to_zhtw(text)
        assert "check" in out
        assert "GPT-4, o3.5 models' output" in out
        assert out.startswith("請")

    def test_ascii_only_unchanged(self):
        text = "This is a pure English sentence, version 2.0."
        assert to_zhtw(text) == text

    def test_empty_string(self):
        assert to_zhtw("") == ""

    def test_traditional_input_stable_chars(self):
        # Already-Traditional characters stay Traditional.
        out = to_zhtw("會議")
        assert "會議" in out


class TestNormalizeRecord:
    def _record(self) -> MeetingRecord:
        return MeetingRecord(
            audio_path="a.wav",
            duration_s=10.0,
            sample_rate=24000,
            language="zh-TW-en",
            source="test",
            split="train",
            segments=[
                Segment(0.0, 3.0, "0", "这个软件很好用"),
                Segment(3.0, 6.0, "1", "we should sync offline"),
            ],
            meta={"k": "v"},
        )

    def test_all_segments_normalized(self):
        rec = normalize_record(self._record())
        assert "軟體" in rec.segments[0].text
        assert "这" not in rec.segments[0].text
        assert rec.segments[1].text == "we should sync offline"

    def test_input_not_mutated_and_fields_preserved(self):
        original = self._record()
        rec = normalize_record(original)
        assert original.segments[0].text == "这个软件很好用"
        assert rec is not original
        assert (rec.audio_path, rec.duration_s, rec.sample_rate) == ("a.wav", 10.0, 24000)
        assert (rec.language, rec.source, rec.split) == ("zh-TW-en", "test", "train")
        assert rec.meta == {"k": "v"}
        assert rec.segments[0].start == 0.0 and rec.segments[0].speaker == "0"
