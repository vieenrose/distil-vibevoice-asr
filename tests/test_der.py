"""Tests for frame-based diarization error rate (DER)."""

from __future__ import annotations

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.der import der


def seg(start: float, end: float, speaker: str) -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text="x")


def fixture_meeting() -> list[Segment]:
    """Tiny 3-speaker meeting (same layout as the cpWER fixture)."""
    return [
        seg(0.0, 3.0, "A"),
        seg(3.0, 6.5, "B"),
        seg(6.5, 10.0, "C"),
        seg(10.0, 13.0, "A"),
        seg(13.0, 16.0, "B"),
    ]


def test_der_identical_is_zero() -> None:
    ref = fixture_meeting()
    assert der(ref, fixture_meeting()) == 0.0


def test_der_identical_zero_with_relabeled_speakers() -> None:
    ref = fixture_meeting()
    hyp = [Segment(s.start, s.end, "spk" + s.speaker, s.text) for s in ref]
    assert der(ref, hyp) == 0.0


def test_der_half_missed() -> None:
    ref = [seg(0.0, 10.0, "A")]
    hyp = [seg(0.0, 5.0, "A")]
    assert abs(der(ref, hyp, collar=0.0) - 0.5) < 0.02


def test_der_all_missed_is_one() -> None:
    ref = [seg(0.0, 10.0, "A")]
    assert der(ref, [], collar=0.0) == 1.0


def test_der_false_alarm() -> None:
    ref = [seg(0.0, 10.0, "A")]
    hyp = [seg(0.0, 12.0, "A")]  # 2s of false alarm outside any collar
    assert abs(der(ref, hyp, collar=0.0) - 0.2) < 0.02


def test_der_collar_excludes_boundary_errors() -> None:
    ref = [seg(0.0, 10.0, "A")]
    hyp = [seg(0.2, 10.0, "A")]  # 0.2s late start, within the 0.25s collar
    assert der(ref, hyp, collar=0.25) == 0.0
    assert der(ref, hyp, collar=0.0) > 0.0


def test_der_speaker_confusion() -> None:
    ref = [seg(0.0, 5.0, "A"), seg(5.0, 10.0, "B")]
    hyp = [seg(0.0, 10.0, "A")]  # second half attributed to the wrong speaker
    assert abs(der(ref, hyp, collar=0.0) - 0.5) < 0.02


def test_der_overlapping_speech_missed_speaker() -> None:
    # Two ref speakers talk simultaneously; hyp only finds one.
    ref = [seg(0.0, 10.0, "A"), seg(0.0, 10.0, "B")]
    hyp = [seg(0.0, 10.0, "A")]
    assert abs(der(ref, hyp, collar=0.0) - 0.5) < 0.02


def test_der_overlapping_speech_both_found_is_zero() -> None:
    ref = [seg(0.0, 10.0, "A"), seg(0.0, 10.0, "B")]
    hyp = [seg(0.0, 10.0, "x"), seg(0.0, 10.0, "y")]
    assert der(ref, hyp, collar=0.0) == 0.0


def test_der_optimal_mapping_not_first_come() -> None:
    # Hyp labels are swapped relative to ref; optimal mapping makes DER 0.
    ref = [seg(0.0, 5.0, "A"), seg(5.0, 10.0, "B")]
    hyp = [seg(0.0, 5.0, "B"), seg(5.0, 10.0, "A")]
    assert der(ref, hyp, collar=0.0) == 0.0


def test_der_empty_inputs() -> None:
    assert der([], []) == 0.0
    # No ref speech, some hyp speech -> worst case 1.0.
    assert der([], [seg(0.0, 5.0, "A")], collar=0.0) == 1.0
