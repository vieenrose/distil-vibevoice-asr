#!/usr/bin/env python
"""Quantization sweep across MULTIPLE clips, scored against official PyTorch f32.

Single-clip comparisons produced a non-monotonic ranking (q5 > f16 > q8 > iq4_xs
on dense180), which is not a quality gradient -- it is coin-flips on near-tied
decisions arranged in a plausible-looking order. Two conclusions today were
already wrong for exactly this reason ("q5 is f16-identical"; "v7-q4 destroys
diarization while f16 preserves it", where v7.2's q4 later beat its own f16).

So: several clips, several quantizations, aggregate before concluding. Ground
truth is the official HF pipeline in float32 for each clip.

Reported per (clip, quant): character differences from official with punctuation
stripped (recognition), segment count (structure), and whether any difference is
a multi-character substitution (substantive) rather than a homophone or a single
dropped character (cosmetic).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = "/home/luigi/RapidSpeech.cpp/build-cuda/moss-td-test"
PUNCT = r"[，。、！？；：,.!?;:\s]"
TMP = Path("/tmp/claude-1001")


def strip(s: str, punct: bool = True) -> str:
    s = re.sub(r"\[\d+(?:\.\d+)?\]", "", s)
    s = re.sub(r"\[S\d+\]", "", s)
    return re.sub(PUNCT, "", s) if punct else s


def nsegs(s: str) -> int:
    return len(re.findall(r"\[\d+(?:\.\d+)?\]", s)) // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="clip stems present in /tmp/claude-1001 as <stem>.wav")
    ap.add_argument("--quants", nargs="+",
                    default=["base_f16", "base_q8_0", "base_q5", "base_q4", "base_iq4_xs"])
    ap.add_argument("--eviction", default="45")
    ap.add_argument("--out", default="/tmp/claude-1001/quant_sweep.json")
    args = ap.parse_args()

    import os
    results: dict = {}
    for clip in args.clips:
        off_p = TMP / f"official_{clip}.txt"
        if not off_p.exists():
            print(f"[skip] {clip}: no official reference"); continue
        official = off_p.read_text(encoding="utf-8").rstrip("\n")
        o = strip(official)
        results[clip] = {"official_segs": nsegs(official), "official_chars": len(o)}
        for q in args.quants:
            gguf = TMP / f"{q}.gguf"
            if not gguf.exists():
                continue
            env = dict(os.environ)
            env["RS_AUDIO_KV_WINDOW"] = args.eviction
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/targets/x86_64-linux/lib"
            try:
                out = subprocess.run([BIN, str(gguf), str(TMP / f"{clip}.wav"), "--gpu"],
                                     capture_output=True, text=True, timeout=6000,
                                     env=env).stdout
            except Exception as e:
                results[clip][q] = {"error": str(e)[:40]}; continue
            line = [l for l in out.splitlines() if "Qwen3ASR: " in l]
            gen = line[-1].split("Qwen3ASR: ", 1)[1] if line else ""
            e = strip(gen)
            sm = difflib.SequenceMatcher(None, o, e, autojunk=False)
            diffs = [(i1, o[i1:i2], e[j1:j2])
                     for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
            # substantive = a replaced span of 2+ chars on either side
            subst = sum(1 for _, a, b in diffs if len(a) >= 2 or len(b) >= 2)
            results[clip][q] = {"ndiff": len(diffs), "substantive": subst,
                                "segs": nsegs(gen), "chars": len(e),
                                "ratio": round(sm.ratio(), 5),
                                "examples": [[a, b] for _, a, b in diffs[:4]]}
            print(f"{clip:16s} {q:14s} diffs={len(diffs):3d} subst={subst:2d} "
                  f"segs={nsegs(gen):3d} (official {results[clip]['official_segs']}) "
                  f"ratio={sm.ratio():.4f}", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n=== AGGREGATE over {len(results)} clips ===")
    print(f"{'quant':14s} {'tot diffs':>10s} {'substantive':>12s} {'mean ratio':>11s} {'seg ratio':>10s}")
    for q in args.quants:
        rows = [results[c][q] for c in results if q in results[c] and "ndiff" in results[c][q]]
        if not rows:
            continue
        td = sum(r["ndiff"] for r in rows)
        ts = sum(r["substantive"] for r in rows)
        mr = sum(r["ratio"] for r in rows) / len(rows)
        sr = sum(r["segs"] for r in rows) / max(1, sum(results[c]["official_segs"] for c in results if q in results[c]))
        print(f"{q:14s} {td:>10d} {ts:>12d} {mr:>11.5f} {sr:>10.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
