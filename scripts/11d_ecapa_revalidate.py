#!/usr/bin/env python
"""Re-run scripts/11c Part B (18-min recurring-speaker recombination test) with the
ECAPA speaker embedder instead of the MFCC-stats placeholder.

Reuses 11c's ``build_long_recurring`` (same rng seed 7 -> byte-identical meeting)
and ``score``/``run_config`` so the only change is the embedder + registry
match_threshold.  Sweeps a few thresholds around the recommended 0.38 and reports
FULL (registry+consolidation) and NO-CONSOLIDATE consistency / DER / #global speakers,
against the MFCC baseline (FULL 0.197 / 0.829 / 1 spk) and the Part A per-window
ceiling (~0.95).

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python scripts/11d_ecapa_revalidate.py [--thresholds 0.30 0.38 0.45]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_11c():
    spec = importlib.util.spec_from_file_location(
        "eval_11c", str(ROOT / "scripts" / "11c_eval_synthetic.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.38, 0.45])
    args = ap.parse_args()

    c = _load_11c()
    from distil_vibevoice.data.pseudo_label import TeacherLabeler
    from distil_vibevoice.runtime.embeddings import load_embedder
    from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

    OUT = c.OUT
    SR = c.SR

    print("loading teacher...")
    labeler = TeacherLabeler(str(ROOT / "models/teacher"))

    print("building long recurring-speaker meeting (seed 7, identical to 11c)...")
    rng = np.random.default_rng(7)
    wav, ref_segs = c.build_long_recurring(n_speakers=5, target_min=15.0, rng=rng)
    long_path = str(OUT / "long_meeting.wav")
    sf.write(long_path, wav, SR)
    true_spk = len(set(s.speaker for s in ref_segs))
    print(f"  long meeting: {len(wav)/SR/60:.1f} min, {true_spk} true speakers, {len(ref_segs)} turns")

    results: dict = {"true_speakers": true_spk, "long_meeting_min": round(len(wav)/SR/60, 1), "sweep": {}}
    for thr in args.thresholds:
        key = f"thr_{thr:.2f}"
        print(f"\n=== ECAPA @ match_threshold={thr:.2f} ===")
        row = {}
        for name, cons in [("full", True), ("noconsol", False)]:
            emb = load_embedder("ecapa")  # device='cpu', dim 192
            reg = SpeakerRegistry(embed_dim=192, match_threshold=thr)
            hyp = c.run_config(labeler, long_path, f"ecapa_{key}_{name}", emb, reg, cons)
            sc = c.score(ref_segs, hyp)
            row[name] = sc
            print(f"  {name:9s}: true={sc['ref_spk']}spk hyp={sc['hyp_spk']}spk consistency={sc['consistency']} der={sc['der']}")
            del emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["sweep"][key] = row

    (OUT / "results_11d_ecapa.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nwrote", OUT / "results_11d_ecapa.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
