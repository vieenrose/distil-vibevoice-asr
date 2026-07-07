"""Tests for global speaker-identity consistency (CPU-only, deterministic)."""

from __future__ import annotations

import pytest

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval import speaker_consistency
from distil_vibevoice.eval.der import der


def seg(start: float, end: float, speaker: str) -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text="x")


def fixture_meeting() -> list[Segment]:
    return [
        seg(0.0, 3.0, "A"),
        seg(3.0, 6.5, "B"),
        seg(6.5, 10.0, "C"),
        seg(10.0, 13.0, "A"),
        seg(13.0, 16.0, "B"),
    ]


def test_identical_is_one() -> None:
    ref = fixture_meeting()
    assert speaker_consistency(ref, fixture_meeting()) == 1.0


def test_permuted_labels_still_one() -> None:
    ref = fixture_meeting()
    remap = {"A": "spk2", "B": "spk0", "C": "spk1"}
    hyp = [Segment(s.start, s.end, remap[s.speaker], s.text) for s in ref]
    assert speaker_consistency(ref, hyp) == 1.0


def test_hand_computed_time_weighted_drop() -> None:
    # ref: A [0,10], B [10,20].  hyp keeps X for [0,10] but wrongly re-uses X
    # for [15,20] (which is really B) and uses Y for [10,15].
    #   optimal map: X->A (10s overlap > 5s with B), Y->B.
    #   correct hyp time: X∩A = 10s, Y∩B = 5s; wrong: X over [15,20] = 5s.
    #   consistency = 15 / 20 = 0.75.
    ref = [seg(0.0, 10.0, "A"), seg(10.0, 20.0, "B")]
    hyp = [seg(0.0, 10.0, "X"), seg(10.0, 15.0, "Y"), seg(15.0, 20.0, "X")]
    assert speaker_consistency(ref, hyp) == pytest.approx(0.75, abs=1e-3)


def test_catches_midmeeting_identity_swap() -> None:
    # Each ref speaker talks in two blocks; hyp assigns a fresh global id to
    # the second block of each, so no single mapping can be right everywhere.
    ref = [seg(0.0, 10.0, "A"), seg(10.0, 20.0, "A"),
           seg(20.0, 30.0, "B"), seg(30.0, 40.0, "B")]
    hyp = [seg(0.0, 10.0, "P"), seg(10.0, 20.0, "Q"),
           seg(20.0, 30.0, "P"), seg(30.0, 40.0, "Q")]
    # Optimal single map (P->A) leaves half of each speaker's time wrong.
    assert speaker_consistency(ref, hyp) == pytest.approx(0.5, abs=1e-3)


def test_extra_hyp_speaker_lowers_score_by_its_time() -> None:
    # ref is one speaker for 10s; hyp splits the last 4s onto a spurious id.
    ref = [seg(0.0, 10.0, "A")]
    hyp = [seg(0.0, 6.0, "X"), seg(6.0, 10.0, "Z")]
    # Only one hyp id can map to A (X, the larger); Z is unmatched -> wrong.
    assert speaker_consistency(ref, hyp) == pytest.approx(0.6, abs=1e-3)


def test_missing_hyp_speaker_merges_two_refs() -> None:
    # Two ref speakers, hyp uses a single label for both -> the label maps to
    # one of them; the other's time is wrong.
    ref = [seg(0.0, 6.0, "A"), seg(6.0, 10.0, "B")]
    hyp = [seg(0.0, 10.0, "X")]  # X->A (6s > 4s); B's 4s is wrong.
    assert speaker_consistency(ref, hyp) == pytest.approx(0.6, abs=1e-3)


def test_overlapping_speech_is_speaker_time_weighted() -> None:
    # Both speakers talk simultaneously for the whole clip; hyp gets both.
    ref = [seg(0.0, 10.0, "A"), seg(0.0, 10.0, "B")]
    hyp = [seg(0.0, 10.0, "P"), seg(0.0, 10.0, "Q")]
    assert speaker_consistency(ref, hyp) == 1.0


def test_empty_hyp_is_vacuously_consistent() -> None:
    ref = [seg(0.0, 10.0, "A")]
    assert speaker_consistency(ref, []) == 1.0
    assert speaker_consistency([], []) == 1.0


def test_hyp_against_empty_ref_is_zero() -> None:
    assert speaker_consistency([], [seg(0.0, 5.0, "A")]) == 0.0


def test_consistent_relabel_scores_one_like_der() -> None:
    # A whole-meeting consistent relabel is tolerated (same as DER): both
    # perfect.  This is the property that makes a per-frame identity check
    # meaningful only when scored against a single global mapping.
    ref = fixture_meeting()
    remap = {"A": "z9", "B": "z0", "C": "z5"}
    hyp = [Segment(s.start, s.end, remap[s.speaker], s.text) for s in ref]
    assert speaker_consistency(ref, hyp) == 1.0
    assert der(ref, hyp, collar=0.0) == 0.0


def test_deterministic() -> None:
    ref = fixture_meeting()
    hyp = [seg(0.0, 3.0, "X"), seg(3.0, 6.5, "Y"), seg(6.5, 10.0, "X"),
           seg(10.0, 13.0, "X"), seg(13.0, 16.0, "Y")]
    a = speaker_consistency(ref, hyp)
    b = speaker_consistency(ref, hyp)
    assert a == b
