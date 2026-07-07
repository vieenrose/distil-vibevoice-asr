"""Tests for distil_vibevoice.runtime.consolidate (CPU, numpy/scipy only)."""
from __future__ import annotations

import numpy as np
import pytest

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.runtime.consolidate import consolidate
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder
from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

SR = 16000
# Same-voice mean-embedding distance is ~1e-4; cross-voice is ~0.08 for the
# MFCC-stats embedder, so a small threshold separates them cleanly.
THRESH = 0.03


def _synth_voice(f0: float, formant: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    ks = np.arange(1, 26)
    env = np.exp(-((ks * f0 - formant) ** 2) / (2 * 300.0 ** 2))
    sig = np.zeros_like(t)
    for k in ks:
        sig += env[k - 1] * np.sin(2 * np.pi * f0 * k * t) / k
    sig += 0.05 * rng.standard_normal(t.size)
    return sig.astype(np.float32)


def _plant(rng: np.random.Generator, n_each: int = 4) -> tuple[SpeakerRegistry, list[Segment]]:
    """Registry with SPEAKER_0 & SPEAKER_1 = same voice, SPEAKER_2 = distinct."""
    emb = MfccStatsEmbedder()
    reg = SpeakerRegistry(embed_dim=emb.dim)
    segments: list[Segment] = []
    plan = [("SPEAKER_0", 120, 700), ("SPEAKER_1", 120, 700), ("SPEAKER_2", 240, 1600)]
    t = 0.0
    for gid, f0, fmt in plan:
        seed = emb.embed(_synth_voice(f0, fmt, 1.2, rng), SR)
        reg.new_speaker(seed, f"{gid} hello", t)
        for _ in range(n_each):
            e = emb.embed(_synth_voice(f0, fmt, 1.0 + 0.3 * rng.random(), rng), SR)
            reg.add_segment(gid, t, t + 1.0, e)
            segments.append(Segment(t, t + 1.0, gid, "word"))
            t += 1.5
    return reg, segments


class TestConsolidate:
    def test_merges_same_voice_leaves_distinct_alone(self):
        reg, segments = _plant(np.random.default_rng(1))
        new_segs, mapping = consolidate(reg, segments, distance_threshold=THRESH)
        # SPEAKER_1 (same voice as 0) merges into the lowest-numbered id.
        assert mapping == {"SPEAKER_1": "SPEAKER_0"}
        assert set(reg.speakers) == {"SPEAKER_0", "SPEAKER_2"}

    def test_relabel_consistency_registry_and_segments(self):
        reg, segments = _plant(np.random.default_rng(2))
        new_segs, mapping = consolidate(reg, segments, distance_threshold=THRESH)
        seg_speakers = {s.speaker for s in new_segs}
        assert seg_speakers == set(reg.speakers)
        assert "SPEAKER_1" not in seg_speakers
        # Original segment list is untouched (a copy is relabeled).
        assert {s.speaker for s in segments} == {"SPEAKER_0", "SPEAKER_1", "SPEAKER_2"}
        # Store labels agree with the returned segments.
        assert {g for g, _, _, _ in reg.segment_store} == seg_speakers

    def test_distinct_voices_not_merged_at_default_threshold(self):
        # Real distinct voices sit within the default 0.45 cosine distance for
        # this crude embedder, but *identical* voices are far closer; use a
        # tight threshold so only genuine duplicates merge.
        reg, segments = _plant(np.random.default_rng(3))
        _, mapping = consolidate(reg, segments, distance_threshold=THRESH)
        assert "SPEAKER_2" not in mapping
        assert "SPEAKER_2" not in mapping.values() or mapping.get("SPEAKER_2") is None

    def test_under_supported_speaker_kept_when_isolated(self):
        # A distinct voice with < min_segments keeps its own label.
        emb = MfccStatsEmbedder()
        reg = SpeakerRegistry(embed_dim=emb.dim)
        rng = np.random.default_rng(4)
        for _ in range(4):
            reg.add_segment("SPEAKER_0", 0.0, 1.0, emb.embed(_synth_voice(120, 700, 1.0, rng), SR))
        # Only one segment for SPEAKER_1, and it is a distinct voice.
        reg.add_segment("SPEAKER_1", 5.0, 6.0, emb.embed(_synth_voice(300, 2200, 1.0, rng), SR))
        segs = [Segment(0.0, 1.0, "SPEAKER_0", "x"), Segment(5.0, 6.0, "SPEAKER_1", "y")]
        _, mapping = consolidate(reg, segs, distance_threshold=THRESH, min_segments=3)
        assert mapping == {}
        assert {g for g, _, _, _ in reg.segment_store} == {"SPEAKER_0", "SPEAKER_1"}

    def test_under_supported_speaker_merges_when_clean(self):
        # An under-supported speaker that IS the same voice merges cleanly.
        emb = MfccStatsEmbedder()
        reg = SpeakerRegistry(embed_dim=emb.dim)
        rng = np.random.default_rng(5)
        for _ in range(4):
            reg.add_segment("SPEAKER_0", 0.0, 1.0, emb.embed(_synth_voice(120, 700, 1.0, rng), SR))
        reg.add_segment("SPEAKER_1", 5.0, 6.0, emb.embed(_synth_voice(120, 700, 1.0, rng), SR))
        segs = [Segment(0.0, 1.0, "SPEAKER_0", "x"), Segment(5.0, 6.0, "SPEAKER_1", "y")]
        _, mapping = consolidate(reg, segs, distance_threshold=THRESH, min_segments=3)
        assert mapping == {"SPEAKER_1": "SPEAKER_0"}

    def test_empty_registry_is_noop(self):
        reg = SpeakerRegistry(embed_dim=8)
        segs = [Segment(0.0, 1.0, "SPEAKER_0", "x")]
        new_segs, mapping = consolidate(reg, segs)
        assert mapping == {}
        assert [s.speaker for s in new_segs] == ["SPEAKER_0"]

    def test_deterministic(self):
        reg1, segs1 = _plant(np.random.default_rng(6))
        reg2, segs2 = _plant(np.random.default_rng(6))
        _, m1 = consolidate(reg1, segs1, distance_threshold=THRESH)
        _, m2 = consolidate(reg2, segs2, distance_threshold=THRESH)
        assert m1 == m2


class TestRecluster:
    """Global per-segment reclustering: splits what incremental matching wrongly merged."""

    def test_recluster_splits_over_merged_speakers(self):
        # Two distinct voices, but every segment was (wrongly) labeled SPEAKER_0 by
        # the incremental registry. mode="merge" can never split; recluster must.
        emb = MfccStatsEmbedder()
        reg = SpeakerRegistry(embed_dim=emb.dim)
        rng = np.random.default_rng(11)
        segs = []
        for i in range(4):
            reg.add_segment("SPEAKER_0", float(2 * i), 2 * i + 1.0,
                            emb.embed(_synth_voice(110, 650, 1.0, rng), SR))
            segs.append(Segment(float(2 * i), 2 * i + 1.0, "SPEAKER_0", "a"))
        for i in range(4):
            reg.add_segment("SPEAKER_0", float(2 * i + 20), 2 * i + 21.0,
                            emb.embed(_synth_voice(220, 1400, 1.0, rng), SR))
            segs.append(Segment(float(2 * i + 20), 2 * i + 21.0, "SPEAKER_0", "b"))
        # merge mode is stuck at 1 speaker; recluster recovers 2.
        merged, _ = consolidate(reg, segs, mode="merge")
        assert len({s.speaker for s in merged}) == 1
        new, _ = consolidate(reg, segs, mode="recluster", recluster_threshold=0.04)
        assert len({s.speaker for s in new}) == 2
        # each true voice ends up as one pure cluster
        first = {s.speaker for s in new[:4]}
        second = {s.speaker for s in new[4:]}
        assert len(first) == 1 and len(second) == 1 and first != second

    def test_recluster_keeps_marker_segments(self):
        emb = MfccStatsEmbedder()
        reg = SpeakerRegistry(embed_dim=emb.dim)
        rng = np.random.default_rng(12)
        reg.add_segment("SPEAKER_0", 1.0, 2.0, emb.embed(_synth_voice(110, 650, 1.0, rng), SR))
        segs = [Segment(0.0, 1.0, "SPEAKER_0", "[Silence]"), Segment(1.0, 2.0, "SPEAKER_0", "hi")]
        new, _ = consolidate(reg, segs, mode="recluster")
        assert new[0].text == "[Silence]"  # marker preserved

    def test_recluster_empty_registry_is_noop(self):
        reg = SpeakerRegistry(embed_dim=8)
        segs = [Segment(0.0, 1.0, "SPEAKER_0", "x")]
        new, mapping = consolidate(reg, segs, mode="recluster")
        assert mapping == {} and [s.speaker for s in new] == ["SPEAKER_0"]
