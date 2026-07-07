#!/usr/bin/env python
"""Chunk-and-recombine a real 60-min IVOD meeting with the real VibeVoice-ASR
teacher, three ways, plus a real-audio GPU RAM validation.

Configs (all offset-corrected global timestamps, window=300s/overlap=45s):
  (A) FULL          : MfccStatsEmbedder + SpeakerRegistry(96), consolidate_on_finish=True
  (B) NO-CONSOLIDATE: same registry/embedder, consolidate_on_finish=False (ablation)
  (C) LEGACY        : embedder=None, registry=None (stitch-only continuity)

RAM validation: torch.cuda.max_memory_allocated for encode+generate of ONE
5-min vs ONE 15-min real window (reset_peak between). We report the *activation*
delta (peak - post-load baseline) to compare against the encode-activation
finding (~1.2 GB @5min vs ~3.5 GB @15min), plus the absolute peak.

Outputs manifests to data/eval_long/ and a results JSON alongside.

HONESTY: the counts are "#global speakers each config produced", not a gold
speaker count; MfccStatsEmbedder is a numpy-MFCC placeholder (production would
use ONNX-ECAPA). Numbers reported are measured, not assumed.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch

from distil_vibevoice.data.manifest import write_manifest
from distil_vibevoice.data.pseudo_label import TeacherLabeler
from distil_vibevoice.runtime.chunked_inference import ChunkedTranscriber
from distil_vibevoice.runtime.embeddings import MfccStatsEmbedder
from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

REPO = Path("/home/luigi/distil-vibevoice-asr")
WAV = REPO / "data/raw/ivod_long/ivod_2025_16530.wav"
TEACHER = REPO / "models/teacher"
OUT = REPO / "data/eval_long"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results_11b.json"

GiB = float(1 << 30)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _n_global(rec) -> int:
    return len({s.speaker for s in rec.segments})


# The IVOD wav has a long silent pre-meeting lead-in; the pyannote reference
# shows speech only from ~1720s. Measure RAM on a speech-containing region so
# the "encode+generate" peak actually exercises generation, not just silence.
RAM_START_S = 1700.0


def measure_window_ram(labeler: TeacherLabeler, clip_s: float, sr: int) -> dict:
    """Peak CUDA activation (GiB) for encode+generate of one real speech clip."""
    start = int(RAM_START_S * sr)
    n = int(clip_s * sr)
    data, _ = sf.read(str(WAV), start=start, stop=start + n, dtype="float32")
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(p, data, sr)
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = time.time()
        rec = labeler.label_file(p)
        torch.cuda.synchronize()
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated()
    finally:
        os.remove(p)
    return {
        "clip_s": clip_s,
        "start_s": RAM_START_S,
        "activation_gib": (peak - base) / GiB,
        "peak_gib": peak / GiB,
        "baseline_gib": base / GiB,
        "gen_s": dt,
        "n_segments": len(rec.segments),
    }


def run_config(labeler, name, out_name, embedder, registry, consolidate) -> dict:
    _log(f"=== config {name}: transcribe start ===")
    tr = ChunkedTranscriber(
        labeler,
        window_s=300.0,
        overlap_s=45.0,
        max_roster=12,
        embedder=embedder,
        registry=registry,
        consolidate_on_finish=consolidate,
    )
    t = time.time()
    rec = tr.transcribe(str(WAV))
    dt = time.time() - t
    path = OUT / out_name
    write_manifest([rec], path)
    n = _n_global(rec)
    _log(f"=== config {name}: {n} global speakers, {len(rec.segments)} segs, "
         f"{dt/60:.2f} min wall, -> {path} ===")
    return {
        "name": name,
        "path": str(path),
        "n_global_speakers": n,
        "n_segments": len(rec.segments),
        "walltime_s": dt,
        "walltime_min": dt / 60.0,
        "num_windows": rec.meta.get("num_windows"),
        "duration_s": rec.duration_s,
    }


def main() -> None:
    results: dict = {}
    info = sf.info(str(WAV))
    sr = int(info.samplerate)
    dur_min = info.frames / sr / 60.0
    results["audio_duration_min"] = dur_min
    _log(f"meeting {WAV.name}: {dur_min:.2f} min @ {sr} Hz")

    _log("loading teacher (models/teacher, ~8.7B)...")
    labeler = TeacherLabeler(str(TEACHER))
    _log("teacher loaded.")

    # ---- RAM validation on real audio (do first: also a timing probe) ----
    ram5 = measure_window_ram(labeler, 300.0, sr)
    _log(f"RAM  5-min: activation {ram5['activation_gib']:.2f} GiB, "
         f"peak {ram5['peak_gib']:.2f} GiB, gen {ram5['gen_s']:.1f}s, "
         f"{ram5['n_segments']} segs")
    ram15 = measure_window_ram(labeler, 900.0, sr)
    _log(f"RAM 15-min: activation {ram15['activation_gib']:.2f} GiB, "
         f"peak {ram15['peak_gib']:.2f} GiB, gen {ram15['gen_s']:.1f}s, "
         f"{ram15['n_segments']} segs")
    results["ram_5min"] = ram5
    results["ram_15min"] = ram15
    RESULTS.write_text(json.dumps(results, indent=2))

    # ---- Three transcription configs ----
    cfgA = run_config(
        labeler, "FULL", "hyp_full.jsonl",
        embedder=MfccStatsEmbedder(), registry=SpeakerRegistry(96),
        consolidate=True,
    )
    results["config_full"] = cfgA
    RESULTS.write_text(json.dumps(results, indent=2))

    cfgB = run_config(
        labeler, "NO-CONSOLIDATE", "hyp_noconsol.jsonl",
        embedder=MfccStatsEmbedder(), registry=SpeakerRegistry(96),
        consolidate=False,
    )
    results["config_noconsol"] = cfgB
    RESULTS.write_text(json.dumps(results, indent=2))

    cfgC = run_config(
        labeler, "LEGACY", "hyp_legacy.jsonl",
        embedder=None, registry=None,
        consolidate=True,  # no-op without a registry
    )
    results["config_legacy"] = cfgC
    RESULTS.write_text(json.dumps(results, indent=2))

    thr = dur_min / cfgA["walltime_min"]
    results["audio_min_per_walltime_min_full"] = thr
    RESULTS.write_text(json.dumps(results, indent=2))
    _log(f"throughput (config FULL): {thr:.3f} audio-min / wall-min")
    _log(f"DONE. results -> {RESULTS}")


if __name__ == "__main__":
    main()
