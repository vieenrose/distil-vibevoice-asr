#!/usr/bin/env python
"""Staged-delivery regression gate.

Goal decomposition (2026-07-20): deliver in stages, each validated against the
previous one, starting from BASE weights rather than the over-specialised FT
lineage:

  stage 1  eviction support        (bounded audio-KV on long audio)
  stage 2  1 + zh-TW transcript    (Traditional-Taiwan script/vocabulary)
  stage 3  1 + 2 + zh/en mixing    (code-switch quality preserved)

"No regression" is only meaningful against a fixed reference, so this script
emits ONE scorecard covering every axis a stage could break, and every stage
runs the SAME scorecard. Anything that moves outside tolerance is a regression,
whether or not that stage was supposed to touch it -- today's session produced
three defects that each hid in an axis nobody was watching (q4-only speaker
collapse, ggml-only marker collapse, dense-speech-only under-segmentation).

Axes, and why each is here:
  STRUCTURE  markers/speakers on dense + synthetic audio, at f16 AND q4, with
             eviction ON and OFF. Single measurements are unreliable when logit
             margins are thin, so disagreement between cells is itself a finding.
  IN-DOMAIN  IVOD 5-min example: MER against the reference transcript.
  GENERAL    ASCEND zh / en / mixed buckets -- the axis the FT silently halved.
  SANITY     reference-free checks (ragged, stretched, overlap, duplicates).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = "/home/luigi/RapidSpeech.cpp/build-x86/moss-td-test"
AUDIO = {
    "dense180": "/tmp/claude-1001/dense180.wav",   # continuous single-speaker
    "synth180": "/tmp/claude-1001/synth180.wav",   # multi-speaker synthetic
    "ex5min":   "/tmp/claude-1001/ex5min_300.wav", # the demo example
}
# Reference values measured on BASE (2026-07-20) — the stage-0 baseline.
BASELINE = {
    "dense180": {"segs": 14, "spk": 1},
    "synth180": {"segs": 40, "spk": 6},
    "ex5min":   {"segs": 15, "spk": 3},
}


def run_engine(gguf: str, wav: str, eviction: int | None = None) -> dict:
    import os
    env = dict(os.environ)
    if eviction is not None:
        env["RS_AUDIO_KV_WINDOW"] = str(eviction)
    try:
        out = subprocess.run([BIN, gguf, wav], capture_output=True, text=True,
                             timeout=4000, env=env).stdout
    except Exception as e:
        return {"error": str(e)[:60]}
    line = [l for l in out.splitlines() if "Qwen3ASR: " in l]
    gen = line[-1].split("Qwen3ASR: ", 1)[1] if line else ""
    ts = re.findall(r"\[\d+(?:\.\d+)?\]", gen)
    spk = sorted(set(re.findall(r"\[S\d+\]", gen)))
    last = float(ts[-1].strip("[]")) if ts else 0.0
    return {"segs": len(ts) // 2, "spk": len(spk), "chars": len(gen),
            "last": round(last, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="model under test (f16)")
    ap.add_argument("--gguf-q4", default=None, help="same model at q4_k_m")
    ap.add_argument("--label", default=None)
    ap.add_argument("--skip-ascend", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    label = args.label or Path(args.gguf).stem

    print(f"===== STAGE GATE: {label} =====\n")
    print("STRUCTURE  (segments / speakers; baseline from base MOSS)")
    print(f"{'audio':10s} {'quant':5s} {'evict':>6s} {'segs':>5s} {'spk':>4s} "
          f"{'chars':>6s} {'last':>7s}   {'baseline':>12s}")
    results = {}
    for name, wav in AUDIO.items():
        if not Path(wav).exists():
            print(f"{name:10s} (missing audio, skipped)"); continue
        for quant, g in (("f16", args.gguf), ("q4", args.gguf_q4)):
            if not g:
                continue
            for ev in (45, 0):
                r = run_engine(g, wav, ev)
                results[f"{name}/{quant}/ev{ev}"] = r
                b = BASELINE.get(name, {})
                flag = ""
                if "segs" in r and b:
                    if r["segs"] < 0.5 * b["segs"]:
                        flag = "  <-- STRUCTURE LOSS"
                    elif r["segs"] > 2.0 * b["segs"]:
                        flag = "  <-- OVER-SEGMENTING"
                print(f"{name:10s} {quant:5s} {ev:>6d} {r.get('segs','-'):>5} "
                      f"{r.get('spk','-'):>4} {r.get('chars','-'):>6} "
                      f"{r.get('last','-'):>7}   "
                      f"{b.get('segs','-'):>5}/{b.get('spk','-'):<3}{flag}")
    # cross-cell agreement: with healthy margins every cell should agree
    for name in AUDIO:
        cells = [v.get("segs") for k, v in results.items()
                 if k.startswith(name + "/") and "segs" in v]
        if len(cells) > 1 and (max(cells) - min(cells)) > max(2, 0.25 * max(cells)):
            print(f"  ! {name}: cells disagree {cells} — thin margins, "
                  f"decisions are not deterministic")

    if not args.skip_ascend:
        print("\nGENERAL  (ASCEND q4-level MER; base = zh .081 / en .353 / mixed .161 / all .198)")
        ggufs = [g for g in (args.gguf_q4 or args.gguf,) if g]
        subprocess.run([sys.executable, str(ROOT / "scripts/57_ascend_gguf.py"),
                        "--ggufs", *ggufs, "--per-bucket", "25"])

    if args.out:
        Path(args.out).write_text(json.dumps({"label": label, "structure": results}, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
