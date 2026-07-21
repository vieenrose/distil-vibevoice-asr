#!/usr/bin/env python
"""Stage-1 parity: engine configs vs the OFFICIAL PyTorch f32 pipeline.

Ground truth = MOSS-TD run through the official HF pipeline in float32. Byte
identity against it is not reachable — the ggml CPU port diverges from PyTorch
CUDA at the FIRST timestamp (0.00 vs 0.06) on audio embeddings, with the mel
front-end already excluded, so two independent implementations would have to
agree bit-for-bit through 24 encoder + 28 decoder layers. What IS measurable is
how far each engine configuration sits from that reference, so we can pick the
cheapest one whose deviation is acceptable.

Reported per configuration:
  text MER      recognition deviation, timestamps stripped (the thing users read)
  ts MAE/max    timestamp deviation on aligned segments (the thing that drifts)
  segs / spk    structural agreement
  bytes         exact size, and whether it is byte-identical (it will not be)

Text and timestamps are scored separately on purpose: a config can reproduce
every character while shifting every timestamp, and those are different defects
with different consequences.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = "/home/luigi/RapidSpeech.cpp/build-x86/moss-td-test"


def parse(s: str):
    text = re.sub(r"\[\d+(?:\.\d+)?\]", "", s)
    text = re.sub(r"\[S\d+\]", "", text)
    ts = [float(x) for x in re.findall(r"\[(\d+(?:\.\d+)?)\]", s)]
    spk = re.findall(r"\[S\d+\]", s)
    return text.strip(), ts, spk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dense180")
    ap.add_argument("--official", default=None)
    ap.add_argument("--configs", nargs="+",
                    default=["base_f32:45", "base_f32:0", "base_f16:45",
                             "base_f16:0", "base_q5:45", "base_q4:45"])
    ap.add_argument("--rep-penalty", default="1.0",
                    help="engine default is 1.10; official uses plain greedy")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from opencc import OpenCC
    from distil_vibevoice.eval.mer import mer
    cc = OpenCC("s2t")

    off_path = args.official or f"/tmp/claude-1001/official_{args.clip}.txt"
    official = Path(off_path).read_text(encoding="utf-8").rstrip("\n")
    o_text, o_ts, o_spk = parse(official)
    print(f"GROUND TRUTH: official PyTorch f32 — {off_path}")
    print(f"  {len(official)} bytes | {len(o_text)} text chars | "
          f"{len(o_ts)} timestamps | {len(set(o_spk))} speakers\n")

    print(f"{'config':22s} {'bytes':>7s} {'ident':>6s} {'textMER':>8s} "
          f"{'tsMAE':>7s} {'tsMax':>7s} {'segs':>5s} {'spk':>4s}")
    for cfg in args.configs:
        model, ev = cfg.split(":")
        gguf = f"/tmp/claude-1001/{model}.gguf"
        if not Path(gguf).exists():
            print(f"{cfg:22s} (missing)"); continue
        import os
        env = dict(os.environ)
        env["RS_AUDIO_KV_WINDOW"] = ev
        env["RS_REP_PENALTY"] = args.rep_penalty
        try:
            out = subprocess.run([BIN, gguf, f"/tmp/claude-1001/{args.clip}.wav"],
                                 capture_output=True, text=True, timeout=6000,
                                 env=env).stdout
        except Exception as e:
            print(f"{cfg:22s} ERROR {str(e)[:30]}"); continue
        line = [l for l in out.splitlines() if "Qwen3ASR: " in l]
        gen = line[-1].split("Qwen3ASR: ", 1)[1] if line else ""
        g_text, g_ts, g_spk = parse(gen)
        ident = "YES" if gen.rstrip("\n") == official else "no"
        m = mer(cc.convert(o_text), cc.convert(g_text))
        n = min(len(o_ts), len(g_ts))
        if n:
            d = [abs(a - b) for a, b in zip(o_ts[:n], g_ts[:n])]
            mae, mx = sum(d) / len(d), max(d)
        else:
            mae = mx = float("nan")
        print(f"{cfg:22s} {len(gen):>7d} {ident:>6s} {m:8.4f} "
              f"{mae:7.2f} {mx:7.2f} {len(g_ts)//2:>5d} {len(set(g_spk)):>4d}")
    print(f"\nofficial reference: segs {len(o_ts)//2}, speakers {len(set(o_spk))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
