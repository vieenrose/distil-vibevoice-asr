"""Tests for distil_vibevoice.runtime.speaker_registry (CPU, numpy only)."""
from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.runtime.speaker_registry import SpeakerProfile, SpeakerRegistry


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class TestConstruction:
    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            SpeakerRegistry(embed_dim=0)
        with pytest.raises(ValueError):
            SpeakerRegistry(embed_dim=4, ema=1.0)
        with pytest.raises(ValueError):
            SpeakerRegistry(embed_dim=4, ema=-0.1)

    def test_dim_mismatch_raises(self):
        reg = SpeakerRegistry(embed_dim=4)
        with pytest.raises(ValueError, match="embedding dim"):
            reg.new_speaker(np.ones(3, dtype=np.float32), "hi", 0.0)


class TestMatchFlow:
    def test_new_speaker_allocates_sequential_ids(self):
        reg = SpeakerRegistry(embed_dim=2)
        a = reg.new_speaker(_unit([1.0, 0.0]), "alice", 1.0)
        b = reg.new_speaker(_unit([0.0, 1.0]), "bob", 2.0)
        assert a == "SPEAKER_0"
        assert b == "SPEAKER_1"
        assert set(reg.speakers) == {"SPEAKER_0", "SPEAKER_1"}

    def test_match_returns_best_speaker(self):
        reg = SpeakerRegistry(embed_dim=2, match_threshold=0.6)
        reg.new_speaker(_unit([1.0, 0.0]), "alice", 1.0)
        reg.new_speaker(_unit([0.0, 1.0]), "bob", 2.0)
        gid, score = reg.match(_unit([0.9, 0.1]))
        assert gid == "SPEAKER_0"
        assert score == pytest.approx(np.dot(_unit([0.9, 0.1]), _unit([1.0, 0.0])), abs=1e-6)

    def test_empty_registry_returns_none(self):
        reg = SpeakerRegistry(embed_dim=2)
        gid, score = reg.match(_unit([1.0, 0.0]))
        assert gid is None
        assert score == 0.0

    def test_threshold_boundary(self):
        reg = SpeakerRegistry(embed_dim=2, match_threshold=0.6)
        reg.new_speaker(_unit([1.0, 0.0]), "alice", 1.0)
        # cos = 0.6 exactly -> matches (>= threshold).
        at = np.array([0.6, np.sqrt(1.0 - 0.36)], dtype=np.float32)
        gid, score = reg.match(at)
        assert gid == "SPEAKER_0"
        assert score == pytest.approx(0.6, abs=1e-6)
        # cos just below threshold -> no match.
        below = np.array([0.59, np.sqrt(1.0 - 0.59 ** 2)], dtype=np.float32)
        gid2, score2 = reg.match(below)
        assert gid2 is None
        assert score2 == pytest.approx(0.59, abs=1e-6)

    def test_zero_embedding_never_matches(self):
        reg = SpeakerRegistry(embed_dim=2, match_threshold=0.6)
        reg.new_speaker(_unit([1.0, 0.0]), "alice", 1.0)
        gid, score = reg.match(np.zeros(2, dtype=np.float32))
        assert gid is None
        assert score == 0.0

    def test_roundoff_seed_centroid_stays_inert(self):
        # Regression: a centroid seeded from a vector that cancels to
        # floating-point roundoff (norm ~1e-15) must normalize to zeros, not be
        # amplified into a unit "garbage" centroid that matches everything.
        reg = SpeakerRegistry(embed_dim=3, match_threshold=0.6)
        gid = reg.new_speaker(np.array([1e-16, -1e-16, 5e-17], np.float32), "s", 1.0)
        assert np.all(reg.speakers[gid].centroid == 0.0)
        # An unrelated silent (all-zero) region must not anchor to it.
        who, score = reg.match(np.zeros(3, dtype=np.float32))
        assert who is None
        assert score == 0.0


class TestUpdate:
    def test_ema_centroid_math(self):
        reg = SpeakerRegistry(embed_dim=2, ema=0.9)
        a = _unit([1.0, 0.0])
        b = _unit([0.0, 1.0])
        reg.new_speaker(a, "alice", 1.0)
        reg.update("SPEAKER_0", b, "again", 5.0)
        expected = 0.9 * a + 0.1 * b
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(reg.speakers["SPEAKER_0"].centroid, expected, atol=1e-6)
        assert reg.speakers["SPEAKER_0"].n_updates == 2
        assert reg.speakers["SPEAKER_0"].last_active == 5.0

    def test_update_unknown_raises(self):
        reg = SpeakerRegistry(embed_dim=2)
        with pytest.raises(KeyError):
            reg.update("SPEAKER_9", _unit([1.0, 0.0]), "x", 1.0)

    def test_snippets_capped_at_three(self):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "s0", 0.0)
        for i, s in enumerate(["a", "b", "c", "d"], start=1):
            reg.update("SPEAKER_0", _unit([1.0, 0.0]), s, float(i))
        assert reg.speakers["SPEAKER_0"].snippets == ["b", "c", "d"]


class TestRoster:
    def test_recency_ordering_and_cap(self):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "first", 10.0)
        reg.new_speaker(_unit([0.0, 1.0]), "second", 30.0)
        reg.new_speaker(_unit([1.0, 1.0]), "third", 20.0)
        prompt = reg.roster_prompt(max_speakers=2)
        lines = prompt.split("\n")
        assert lines == ["SPEAKER_1: second", "SPEAKER_2: third"]

    def test_uses_most_recent_snippet(self):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "hello", 1.0)
        reg.update("SPEAKER_0", _unit([1.0, 0.0]), "latest", 2.0)
        assert reg.roster_prompt() == "SPEAKER_0: latest"

    def test_empty_roster(self):
        reg = SpeakerRegistry(embed_dim=2)
        assert reg.roster_prompt() == ""


class TestRelabel:
    def test_merges_profiles_and_store(self):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "a", 1.0)
        reg.new_speaker(_unit([0.9, 0.1]), "b", 5.0)
        reg.add_segment("SPEAKER_0", 0.0, 1.0, _unit([1.0, 0.0]))
        reg.add_segment("SPEAKER_1", 2.0, 3.0, _unit([0.9, 0.1]))
        reg.relabel({"SPEAKER_1": "SPEAKER_0"})
        assert set(reg.speakers) == {"SPEAKER_0"}
        assert reg.speakers["SPEAKER_0"].n_updates == 2
        assert reg.speakers["SPEAKER_0"].last_active == 5.0
        assert {g for g, _, _, _ in reg.segment_store} == {"SPEAKER_0"}
        # Centroid stays L2-normalized after merge.
        assert np.linalg.norm(reg.speakers["SPEAKER_0"].centroid) == pytest.approx(1.0, abs=1e-5)

    def test_missing_keys_are_identity(self):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "a", 1.0)
        reg.relabel({})
        assert set(reg.speakers) == {"SPEAKER_0"}


class TestSaveLoad:
    def test_round_trip_exact(self, tmp_path):
        reg = SpeakerRegistry(embed_dim=3, ema=0.8, match_threshold=0.55)
        reg.new_speaker(_unit([1.0, 2.0, 3.0]), "alice speaks", 4.0)
        reg.new_speaker(_unit([3.0, 2.0, 1.0]), "bob speaks", 8.0)
        reg.update("SPEAKER_0", _unit([0.0, 1.0, 0.0]), "more", 9.0)
        reg.add_segment("SPEAKER_0", 0.0, 1.5, _unit([0.2, 0.3, 0.9]))
        reg.add_segment("SPEAKER_1", 2.0, 3.5, _unit([0.9, 0.1, 0.4]))

        path = tmp_path / "registry"
        reg.save(path)
        loaded = SpeakerRegistry.load(path)

        assert loaded.embed_dim == reg.embed_dim
        assert loaded.ema == reg.ema
        assert loaded.match_threshold == reg.match_threshold
        assert loaded._next_id == reg._next_id
        assert set(loaded.speakers) == set(reg.speakers)
        for gid, prof in reg.speakers.items():
            lp = loaded.speakers[gid]
            # Bitwise-exact centroids.
            assert np.array_equal(lp.centroid, prof.centroid)
            assert lp.centroid.dtype == np.float32
            assert lp.n_updates == prof.n_updates
            assert lp.snippets == prof.snippets
            assert lp.last_active == prof.last_active
        assert len(loaded.segment_store) == len(reg.segment_store)
        for (g1, s1, e1, v1), (g2, s2, e2, v2) in zip(
            loaded.segment_store, reg.segment_store
        ):
            assert (g1, s1, e1) == (g2, s2, e2)
            assert np.array_equal(v1, v2)

    def test_next_id_preserved_after_load(self, tmp_path):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "a", 1.0)
        reg.new_speaker(_unit([0.0, 1.0]), "b", 2.0)
        path = tmp_path / "reg"
        reg.save(path)
        loaded = SpeakerRegistry.load(path)
        assert loaded.new_speaker(_unit([1.0, 1.0]), "c", 3.0) == "SPEAKER_2"

    def test_load_accepts_json_or_npz_suffix(self, tmp_path):
        reg = SpeakerRegistry(embed_dim=2)
        reg.new_speaker(_unit([1.0, 0.0]), "a", 1.0)
        reg.save(tmp_path / "reg")
        for p in (tmp_path / "reg.json", tmp_path / "reg.npz", tmp_path / "reg"):
            assert set(SpeakerRegistry.load(p).speakers) == {"SPEAKER_0"}


class TestDeterminism:
    def test_same_sequence_gives_identical_state(self):
        def build() -> SpeakerRegistry:
            reg = SpeakerRegistry(embed_dim=2, ema=0.85)
            reg.new_speaker(_unit([1.0, 0.0]), "a", 1.0)
            reg.new_speaker(_unit([0.1, 1.0]), "b", 2.0)
            reg.update("SPEAKER_0", _unit([0.5, 0.5]), "x", 3.0)
            reg.update("SPEAKER_1", _unit([0.2, 0.9]), "y", 4.0)
            return reg

        r1, r2 = build(), build()
        for gid in r1.speakers:
            assert np.array_equal(r1.speakers[gid].centroid, r2.speakers[gid].centroid)


def test_speaker_profile_dataclass_defaults():
    prof = SpeakerProfile(global_id="SPEAKER_0", centroid=np.zeros(2, np.float32), n_updates=1)
    assert prof.snippets == []
    assert prof.last_active == 0.0
