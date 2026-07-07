"""Tests for distil_vibevoice.runtime.speaker_stitch (CPU, scipy only)."""
from __future__ import annotations

import pytest

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.runtime.speaker_stitch import _is_duplicate, match_speakers, stitch


def _chunk_a() -> list[Segment]:
    """Chunk covering [0, 60); overlap region with chunk B is [45, 60)."""
    return [
        Segment(0.0, 4.0, "s1", "hello everyone welcome to the meeting"),
        Segment(5.0, 9.0, "s2", "thanks glad to be here"),
        Segment(50.0, 54.0, "s1", "let us review the quarterly roadmap now"),
        Segment(55.0, 59.0, "s2", "sure the roadmap has three milestones"),
    ]


def _chunk_b() -> list[Segment]:
    """Chunk starting at 45.0 (GLOBAL times), labels permuted: b==s1, a==s2."""
    return [
        Segment(50.2, 54.1, "b", "let us review the quarterly roadmap now"),
        Segment(55.1, 58.9, "a", "sure the roadmap has three milestones"),
        Segment(61.0, 65.0, "a", "milestone one is the mobile release"),
        Segment(65.2, 69.0, "a", "milestone two is quantization"),
        Segment(70.0, 74.0, "c", "hi sorry i am late joining now"),
    ]


class TestMatchSpeakers:
    def test_permuted_labels_are_matched_via_overlap_agreement(self):
        mapping = match_speakers(_chunk_a(), _chunk_b(), 45.0, 60.0)
        assert mapping["b"] == "s1"
        assert mapping["a"] == "s2"

    def test_unmatched_new_speaker_gets_fresh_global_id(self):
        mapping = match_speakers(_chunk_a(), _chunk_b(), 45.0, 60.0)
        # "c" never speaks in the overlap -> fresh id, not colliding with s1/s2.
        assert mapping["c"] == "SPEAKER_0"
        assert set(mapping) == {"a", "b", "c"}

    def test_zero_token_agreement_is_not_matched_despite_time_overlap(self):
        prev = [
            Segment(50.0, 54.0, "X", "the quick brown fox jumps"),
            Segment(55.0, 58.0, "Y", "over the lazy dog again"),
        ]
        new = [
            Segment(50.1, 54.2, "n1", "the quick brown fox jumps"),
            Segment(55.0, 58.0, "n2", "totally different words entirely"),
            Segment(58.5, 59.5, "n3", "completely unrelated content here"),
        ]
        mapping = match_speakers(prev, new, 45.0, 60.0)
        assert mapping["n1"] == "X"
        # n2 overlaps Y in time but shares no tokens -> fresh id.
        assert mapping["n2"] == "SPEAKER_0"
        assert mapping["n3"] == "SPEAKER_1"

    def test_fresh_ids_continue_existing_speaker_numbering(self):
        prev = [Segment(50.0, 54.0, "SPEAKER_0", "alpha beta gamma")]
        new = [Segment(56.0, 58.0, "z", "delta epsilon zeta")]
        mapping = match_speakers(prev, new, 45.0, 60.0)
        assert mapping["z"] == "SPEAKER_1"

    def test_empty_prev_gives_all_fresh_ids_in_appearance_order(self):
        new = [
            Segment(3.0, 4.0, "q", "later words"),
            Segment(0.0, 1.0, "p", "first words"),
        ]
        mapping = match_speakers([], new, 0.0, 5.0)
        assert mapping == {"p": "SPEAKER_0", "q": "SPEAKER_1"}


class TestStitch:
    def test_full_stitch_relabels_dedups_and_merges(self):
        result = stitch([_chunk_a(), _chunk_b()], [0.0, 45.0], overlap_s=15.0)

        by_text = {seg.text: seg for seg in result}
        # Chunk 0 speakers canonicalized by first appearance.
        assert by_text["hello everyone welcome to the meeting"].speaker == "SPEAKER_0"
        assert by_text["thanks glad to be here"].speaker == "SPEAKER_1"
        # Permuted chunk-B labels mapped onto the global ids.
        assert by_text["let us review the quarterly roadmap now"].speaker == "SPEAKER_0"
        assert by_text["sure the roadmap has three milestones"].speaker == "SPEAKER_1"
        assert by_text["hi sorry i am late joining now"].speaker == "SPEAKER_2"

        # Duplicates in the overlap appear exactly once, keeping the version
        # from the chunk whose center is closer (chunk B here).
        texts = [seg.text for seg in result]
        assert texts.count("let us review the quarterly roadmap now") == 1
        assert texts.count("sure the roadmap has three milestones") == 1
        assert by_text["let us review the quarterly roadmap now"].start == pytest.approx(50.2)
        assert by_text["sure the roadmap has three milestones"].start == pytest.approx(55.1)

        # Adjacent same-speaker segments with a 0.2 s gap are merged.
        merged = [seg for seg in result if "milestone one" in seg.text]
        assert len(merged) == 1
        assert merged[0].text == (
            "milestone one is the mobile release milestone two is quantization"
        )
        assert merged[0].start == pytest.approx(61.0)
        assert merged[0].end == pytest.approx(69.0)
        assert merged[0].speaker == "SPEAKER_1"

        # 4 chunk-A segments - 2 dropped dups + 5 chunk-B - 1 merge = 6 total.
        assert len(result) == 6
        starts = [seg.start for seg in result]
        assert starts == sorted(starts)

    def test_merge_respects_gap_threshold_and_speaker(self):
        chunk = [
            Segment(0.0, 1.0, "u", "one"),
            Segment(1.1, 2.0, "u", "two"),  # gap 0.1 -> merged
            Segment(2.5, 3.0, "u", "three"),  # gap 0.5 -> not merged
            Segment(3.1, 4.0, "v", "four"),  # other speaker -> not merged
        ]
        result = stitch([chunk], [0.0], overlap_s=5.0)
        assert [seg.text for seg in result] == ["one two", "three", "four"]
        assert result[0].end == pytest.approx(2.0)

    def test_cjk_texts_merge_without_inserted_space(self):
        chunk = [
            Segment(0.0, 1.0, "u", "大家好"),
            Segment(1.1, 2.0, "u", "歡迎參加會議"),
        ]
        result = stitch([chunk], [0.0], overlap_s=5.0)
        assert len(result) == 1
        assert result[0].text == "大家好歡迎參加會議"

    def test_empty_and_mismatched_inputs(self):
        assert stitch([], [], overlap_s=10.0) == []
        with pytest.raises(ValueError):
            stitch([[]], [0.0, 10.0], overlap_s=5.0)

    def test_inputs_are_not_mutated(self):
        a, b = _chunk_a(), _chunk_b()
        stitch([a, b], [0.0, 45.0], overlap_s=15.0)
        assert a[0].speaker == "s1"
        assert b[0].speaker == "b"
        assert len(b) == 5


class TestClippedFragmentDedup:
    """Turns straddling a window boundary must not duplicate their overlap text.

    With window 12 s / overlap 4 s (offsets 0.0 and 8.0), a turn spanning
    6.2-8.8 s is fully transcribed by window 0 and re-transcribed by window 1
    as the clipped fragment 8.0-8.8 s.  Time-IoU of the pair is only
    0.8 / 2.6 = 0.31, so an IoU >= 0.5 gate misses it; containment (1.0)
    catches it and the longer (un-clipped) version must win.
    """

    def test_boundary_clipped_fragment_is_dropped(self):
        chunk0 = [
            Segment(0.0, 4.0, "s0", "intro words spoken here"),
            Segment(6.2, 8.8, "s1", "歡迎大家參加會議 hello"),
        ]
        chunk1 = [
            Segment(8.0, 8.8, "x", "會議 hello"),  # clipped re-transcription
            Segment(9.5, 11.0, "y", "next topic starts now"),
        ]
        result = stitch([chunk0, chunk1], [0.0, 8.0], overlap_s=4.0)

        texts = [seg.text for seg in result]
        assert "歡迎大家參加會議 hello" in texts
        # The fragment is deduped, not merged back in: no doubled words.
        assert " ".join(texts).count("hello") == 1
        assert " ".join(texts).count("會議") == 1
        full = next(seg for seg in result if "歡迎" in seg.text)
        assert full.start == pytest.approx(6.2)
        assert full.end == pytest.approx(8.8)

    def test_unclipped_version_from_new_chunk_replaces_fragment(self):
        # Reverse case: window 0 clips the HEAD of a turn at its window end
        # (12.0 s); window 1 has the full transcription, which must replace it.
        chunk0 = [
            Segment(0.0, 4.0, "s0", "intro words spoken here"),
            Segment(11.5, 12.0, "s1", "下一個"),  # clipped at window end
        ]
        chunk1 = [
            Segment(11.5, 14.0, "a", "下一個議題是預算 planning"),
        ]
        result = stitch([chunk0, chunk1], [0.0, 8.0], overlap_s=4.0)

        texts = [seg.text for seg in result]
        assert "下一個議題是預算 planning" in texts
        assert " ".join(texts).count("下一個") == 1
        full = next(seg for seg in result if "議題" in seg.text)
        assert full.end == pytest.approx(14.0)
        assert full.speaker == "SPEAKER_1"  # mapped onto the global roster

    def test_is_duplicate_uses_containment_and_text_containment(self):
        full = Segment(6.2, 8.8, "A", "歡迎大家參加會議 hello")
        # Same speaker: temporal containment alone suffices.
        assert _is_duplicate(full, Segment(8.0, 8.8, "A", "會議 hello"))
        # Different label but fragment text contained in the full text
        # (token-jaccard is only 3/9, below the jaccard gate).
        assert _is_duplicate(full, Segment(8.0, 8.8, "B", "會議 hello"))
        # Different label and unrelated text: not a duplicate.
        assert not _is_duplicate(full, Segment(8.0, 8.8, "B", "完全不同的內容"))
        # Low temporal containment: not a duplicate even with identical text.
        assert not _is_duplicate(
            Segment(0.0, 2.0, "A", "same words here"),
            Segment(1.9, 4.0, "A", "same words here"),
        )
