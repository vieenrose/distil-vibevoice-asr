#!/usr/bin/env python
"""Diarization eval on CLEAN synthetic meetings with EXACT ground-truth labels.

Closes the loop on docs/LONG_MEETING_EVAL.md: the IVOD (far-field parliamentary)
collapse was traced to the teacher, not the chunk-and-recombine pipeline. Here we
run the same machinery on clean synthetic multi-speaker meetings (built from distinct
Common Voice zh-TW speakers, so ground truth is exact) to confirm the pipeline
recovers high speaker-consistency when the teacher CAN diarize.

Part A: existing short sim meetings (single window) -> teacher raw diarization quality.
Part B: one long (~15 min) recurring-speaker meeting, forced short window -> tests the
        cross-window recombination (FULL / NO-CONSOLIDATE / LEGACY).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from distil_vibevoice.data.manifest import Segment, read_manifest
from distil_vibevoice.data.pseudo_label import TeacherLabeler
from distil_vibevoice.data.simulate_meetings import simulate_meeting
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.der import der
from distil_vibevoice.runtime.chunked_inference import ChunkedTranscriber
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder
from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
OUT = ROOT / "data/eval_synth"
OUT.mkdir(parents=True, exist_ok=True)


def score(ref: list[Segment], hyp: list[Segment]) -> dict:
    ref_spk = len(set(s.speaker for s in ref))
    hyp_spk = len(set(s.speaker for s in hyp))
    try:
        cons = round(speaker_consistency(ref, hyp), 3)
    except Exception as e:  # noqa: BLE001
        cons = f"err:{e}"
    try:
        d = round(der(ref, hyp), 3)
    except Exception as e:  # noqa: BLE001
        d = f"err:{e}"
    return {"ref_spk": ref_spk, "hyp_spk": hyp_spk, "consistency": cons, "der": d}


def _resample(x: np.ndarray, sr_from: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(1)
    if sr_from == SR:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, SR)
    return resample_poly(x, SR // g, sr_from // g).astype(np.float32)


def build_client_index(min_clips: int) -> dict[str, list[str]]:
    tsv = next(iter(glob.glob(str(ROOT / "data/raw/common_voice_zhtw/**/test.tsv"), recursive=True)), None)
    mp3s = glob.glob(str(ROOT / "data/raw/common_voice_zhtw/**/*.mp3"), recursive=True)
    on_disk = {os.path.basename(p): p for p in mp3s}
    by_client: dict[str, list[str]] = defaultdict(list)
    if tsv:
        import csv

        with open(tsv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                name = os.path.basename(row.get("path", ""))
                if name in on_disk:
                    by_client[row["client_id"]].append(on_disk[name])
    return {c: v for c, v in by_client.items() if len(v) >= min_clips}


def build_long_recurring(n_speakers: int, target_min: float, rng: np.random.Generator) -> tuple[np.ndarray, list[Segment]]:
    """Round-robin clips from n distinct real speakers over ~target_min minutes.

    Each speaker recurs across the whole timeline, so with a short transcription
    window every speaker appears in multiple windows -> exercises re-combination.
    """
    idx = build_client_index(min_clips=8)
    clients = sorted(idx, key=lambda c: -len(idx[c]))[:n_speakers]
    utts: list[tuple[np.ndarray, str, str]] = []
    pos = {c: 0 for c in clients}
    total = 0.0
    while total < target_min * 60:
        for si, c in enumerate(clients):
            if pos[c] >= len(idx[c]):
                pos[c] = 0
            wav, sr = sf.read(idx[c][pos[c]])
            pos[c] += 1
            w = _resample(np.asarray(wav), sr)
            if w.size < SR // 2:
                continue
            utts.append((w, str(si), ""))
            total += w.size / SR
        if total >= target_min * 60:
            break
    wav, segs = simulate_meeting(utts, SR, overlap_ratio=0.05, rng=rng)
    return wav, segs


def run_config(labeler, wav_path: str, name: str, embedder, registry, consolidate: bool) -> list[Segment]:
    tr = ChunkedTranscriber(
        labeler,
        window_s=120.0,
        overlap_s=20.0,
        embedder=embedder,
        registry=registry,
        consolidate_on_finish=consolidate,
    )
    rec = tr.transcribe(wav_path)
    (OUT / f"long_{name}.jsonl").write_text(
        json.dumps({**rec.__dict__, "segments": [s.__dict__ for s in rec.segments]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [s for s in rec.segments if not s.text.startswith("[")]


def main() -> int:
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    print("loading teacher...")
    labeler = TeacherLabeler(str(ROOT / "models/teacher"))
    results: dict = {}

    # ---- Part A: existing short sim meetings (single window), clean multi-speaker ----
    sim = [r for r in read_manifest(str(ROOT / "data/manifests/simulated.jsonl"))]
    sim = sorted(sim, key=lambda r: -len(set(s.speaker for s in r.segments)))[:5]
    partA = []
    for r in sim:
        hyp_rec = labeler.label_file(r.audio_path)
        hyp = [s for s in hyp_rec.segments if not s.text.startswith("[")]
        s = score(r.segments, hyp)
        s["meeting"] = os.path.basename(r.audio_path)
        s["dur_s"] = round(r.duration_s)
        partA.append(s)
        print(f"  A {s['meeting']}: true={s['ref_spk']}spk hyp={s['hyp_spk']}spk consistency={s['consistency']} der={s['der']}")
        torch.cuda.empty_cache()
    results["partA_short_singlewindow"] = partA

    # ---- Part B: long recurring-speaker meeting, forced short window ----
    print("building long recurring-speaker meeting...")
    rng = np.random.default_rng(7)
    wav, ref_segs = build_long_recurring(n_speakers=5, target_min=15.0, rng=rng)
    long_path = str(OUT / "long_meeting.wav")
    sf.write(long_path, wav, SR)
    print(f"  long meeting: {len(wav)/SR/60:.1f} min, {len(set(s.speaker for s in ref_segs))} true speakers, {len(ref_segs)} turns")
    partB = {}
    for name, emb, reg, cons in [
        ("full", MfccStatsEmbedder(), SpeakerRegistry(96), True),
        ("noconsol", MfccStatsEmbedder(), SpeakerRegistry(96), False),
        ("legacy", None, None, False),
    ]:
        hyp = run_config(labeler, long_path, name, emb, reg, cons)
        partB[name] = score(ref_segs, hyp)
        print(f"  B {name:9s}: true={partB[name]['ref_spk']}spk hyp={partB[name]['hyp_spk']}spk consistency={partB[name]['consistency']} der={partB[name]['der']}")
        torch.cuda.empty_cache()
    results["partB_long_multiwindow"] = partB
    results["long_meeting_min"] = round(len(wav) / SR / 60, 1)

    (OUT / "results_11c.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "results_11c.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
