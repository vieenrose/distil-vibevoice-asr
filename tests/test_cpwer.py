"""Tests for cpWER (concatenated minimum-permutation word error rate)."""

from __future__ import annotations

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.cpwer import cpwer, optimal_speaker_map


def fixture_meeting() -> list[Segment]:
    """Tiny 3-speaker zh+en code-switched meeting used across eval tests."""
    return [
        Segment(start=0.0, end=3.0, speaker="A", text="大家好 歡迎參加今天的會議"),
        Segment(start=3.0, end=6.5, speaker="B", text="we should review the roadmap first"),
        Segment(start=6.5, end=10.0, speaker="C", text="好的 我來報告 progress"),
        Segment(start=10.0, end=13.0, speaker="A", text="thanks 請開始"),
        Segment(start=13.0, end=16.0, speaker="B", text="第一個 milestone 已經完成"),
    ]


def relabel(segments: list[Segment], mapping: dict[str, str]) -> list[Segment]:
    return [
        Segment(start=s.start, end=s.end, speaker=mapping[s.speaker], text=s.text)
        for s in segments
    ]


def test_cpwer_perfect_is_zero() -> None:
    ref = fixture_meeting()
    assert cpwer(ref, fixture_meeting()) == 0.0


def test_cpwer_invariant_to_speaker_label_permutation() -> None:
    ref = fixture_meeting()
    hyp = relabel(fixture_meeting(), {"A": "spk_2", "B": "spk_0", "C": "spk_1"})
    assert cpwer(ref, hyp) == 0.0


def test_cpwer_catches_attribution_swap() -> None:
    # Same words, but two speakers' segments swapped -> errors > 0.
    ref = fixture_meeting()
    hyp = fixture_meeting()
    hyp[0] = Segment(hyp[0].start, hyp[0].end, "B", hyp[0].text)
    hyp[1] = Segment(hyp[1].start, hyp[1].end, "A", hyp[1].text)
    assert cpwer(ref, hyp) > 0.0


def test_cpwer_word_errors_counted() -> None:
    ref = [Segment(0.0, 2.0, "A", "你好 世界"), Segment(2.0, 4.0, "B", "hello world")]
    hyp = [Segment(0.0, 2.0, "A", "你好 世界"), Segment(2.0, 4.0, "B", "hello word")]
    # 1 wrong en word / 6 ref tokens (4 zh chars + 2 en words).
    assert abs(cpwer(ref, hyp) - 1.0 / 6.0) < 1e-12


def test_cpwer_missing_speaker_counts_as_deletions() -> None:
    ref = fixture_meeting()
    hyp = [s for s in fixture_meeting() if s.speaker != "C"]
    from distil_vibevoice.eval.mer import tokenize_mixed

    ref_c_tokens = len(tokenize_mixed("好的 我來報告 progress"))
    total_ref = sum(len(tokenize_mixed(s.text)) for s in ref)
    assert abs(cpwer(ref, hyp) - ref_c_tokens / total_ref) < 1e-12


def test_cpwer_extra_hyp_speaker_counts_as_insertions() -> None:
    ref = fixture_meeting()
    hyp = fixture_meeting() + [Segment(16.0, 18.0, "D", "totally hallucinated words")]
    assert cpwer(ref, hyp) > 0.0


def test_cpwer_oversegmented_speaker_split() -> None:
    # One ref speaker split into two hyp speakers: the matched hyp stream
    # yields 2 deletions and the unmatched (padded against an empty ref
    # stream) yields 2 insertions -> 4 edits / 4 ref tokens = 1.0.
    ref = [Segment(0.0, 4.0, "A", "one two three four")]
    hyp = [
        Segment(0.0, 2.0, "x", "one two"),
        Segment(2.0, 4.0, "y", "three four"),
    ]
    assert abs(cpwer(ref, hyp) - 1.0) < 1e-12


def test_cpwer_empty_edge_cases() -> None:
    ref = fixture_meeting()
    assert cpwer(ref, []) == 1.0  # everything deleted
    assert cpwer([], []) == 0.0
    assert cpwer([], ref) == 1.0


def test_optimal_speaker_map_recovers_permutation() -> None:
    ref = fixture_meeting()
    hyp = relabel(fixture_meeting(), {"A": "1", "B": "2", "C": "0"})
    assert optimal_speaker_map(ref, hyp) == {"1": "A", "2": "B", "0": "C"}
