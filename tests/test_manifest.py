"""Tests for distil_vibevoice.data.manifest (JSONL IO + teacher format)."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from distil_vibevoice.data.manifest import (
    MeetingRecord,
    Segment,
    format_target,
    iter_manifest,
    parse_teacher_output,
    read_manifest,
    write_manifest,
)

# Real raw output example from the VibeVoice-ASR-HF model card.
_REAL_TEACHER_OUTPUT = (
    "<|im_start|>assistant\n"
    '[{"Start":0,"End":15.43,"Speaker":0,"Content":"Hello everyone and welcome to the '
    "Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the "
    "biggest debates in all of sports: who's the greatest basketball player of all "
    'time? I\'m so excited to have Sam here to talk about it with me."},'
    '{"Start":15.43,"End":21.05,"Speaker":1,"Content":"Thanks so much for having me, '
    'Alex. And you\'re absolutely right. This question always brings out some '
    'seriously strong feelings."},'
    '{"Start":21.05,"End":31.66,"Speaker":0,"Content":"Okay, so let\'s get right into '
    'it. For me, it has to be Michael Jordan. Six trips to the finals, six '
    'championships. That kind of perfection is just incredible."},'
    '{"Start":31.66,"End":40.93,"Speaker":1,"Content":"Oh man, the first thing that '
    "always pops into my head is that shot against the Cleveland Cavaliers back in "
    '\'89. Jordan just rises, hangs in the air forever, and just sinks it."}]'
    "<|im_end|>"
)


def _sample_records() -> list[MeetingRecord]:
    return [
        MeetingRecord(
            audio_path="audio/mtg_001.wav",
            duration_s=123.5,
            sample_rate=24000,
            language="zh-TW-en",
            source="synthetic",
            split="train",
            segments=[
                Segment(0.0, 4.2, "0", "大家好，歡迎參加今天的會議。"),
                Segment(4.2, 9.7, "1", "我們先 review 一下上週的 action items。"),
            ],
            meta={"seed": 42},
        ),
        MeetingRecord(
            audio_path="audio/mtg_002.wav",
            duration_s=61.0,
            sample_rate=16000,
            language="en",
            source="ami",
            split="dev",
            segments=[Segment(1.0, 2.5, "spk_a", "Let's get started.")],
        ),
    ]


class TestManifestIO:
    def test_roundtrip_unicode(self, tmp_path):
        records = _sample_records()
        path = tmp_path / "manifest.jsonl"
        write_manifest(records, path)
        back = read_manifest(path)
        assert back == records
        # Unicode must survive verbatim (ensure_ascii=False).
        raw = path.read_text(encoding="utf-8")
        assert "歡迎參加" in raw

    def test_iter_matches_read(self, tmp_path):
        records = _sample_records()
        path = tmp_path / "manifest.jsonl"
        write_manifest(records, path)
        assert list(iter_manifest(path)) == read_manifest(path)

    def test_malformed_lines_skipped(self, tmp_path):
        records = _sample_records()
        path = tmp_path / "manifest.jsonl"
        write_manifest(records, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines.insert(1, "{not valid json")
        lines.insert(0, '{"audio_path": "x.wav"}')  # valid JSON, missing keys
        lines.append("")  # blank line
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert read_manifest(path) == records

    def test_meta_defaults_to_empty_dict(self, tmp_path):
        rec = _sample_records()[1]
        assert rec.meta == {}
        path = tmp_path / "m.jsonl"
        write_manifest([rec], path)
        assert read_manifest(path)[0].meta == {}


class TestParseTeacherOutput:
    def test_real_hf_example(self):
        segs = parse_teacher_output(_REAL_TEACHER_OUTPUT)
        assert len(segs) == 4
        assert segs[0].start == 0.0
        assert segs[0].end == 15.43
        assert segs[0].speaker == "0"
        assert segs[0].text.startswith("Hello everyone and welcome")
        assert segs[3].speaker == "1"
        assert segs[3].end == 40.93

    def test_original_package_key_spellings(self):
        text = (
            '[{"Start time": 1.0, "End time": 2.5, "Speaker ID": 3, '
            '"Content": "測試 segment 一"}]'
        )
        segs = parse_teacher_output(text)
        assert segs == [Segment(1.0, 2.5, "3", "測試 segment 一")]

    def test_garbage_returns_empty(self):
        assert parse_teacher_output("total nonsense, no json here") == []
        assert parse_teacher_output("") == []

    def test_truncated_output_recovers_prefix(self):
        # Simulates generation cut off mid-array: valid objects recovered.
        text = (
            '[{"Start":0,"End":1.5,"Speaker":0,"Content":"完整的第一段"},'
            '{"Start":1.5,"End":3.0,"Speaker":1,"Content":"完整的第二段"},'
            '{"Start":3.0,"End":4.0,"Speaker":0,"Content":"被截斷的'
        )
        segs = parse_teacher_output(text)
        assert len(segs) == 2
        assert segs[1].text == "完整的第二段"

    def test_malformed_segment_skipped(self):
        text = (
            '[{"Start":0,"End":1.0,"Speaker":0,"Content":"ok"},'
            '{"Speaker":1,"Content":"missing times"},'
            '{"Start":"not a number","End":2.0,"Speaker":0,"Content":"bad start"}]'
        )
        segs = parse_teacher_output(text)
        assert len(segs) == 1
        assert segs[0].text == "ok"


class TestFormatTarget:
    def test_parse_format_inverse(self):
        segments = [
            Segment(0.0, 15.43, "0", "大家好，我是主持人。"),
            Segment(15.43, 21.05, "1", 'He said "quote" and 中文 mixed.'),
            Segment(21.05, 31.66, "2", "下一個 milestone 是 Q3。"),
        ]
        assert parse_teacher_output(format_target(segments)) == segments

    def test_numeric_speakers_emitted_as_ints(self):
        out = format_target([Segment(0.0, 1.0, "7", "hi")])
        assert '"Speaker":7' in out

    def test_non_numeric_speaker_preserved(self):
        segs = [Segment(0.0, 1.0, "spk_a", "hello")]
        assert parse_teacher_output(format_target(segs)) == segs

    def test_empty(self):
        assert format_target([]) == "[]"
        assert parse_teacher_output("[]") == []
