#!/usr/bin/env python3
"""Stage-3 profiler: quantisation ladder on CUDA.

Reports, per (model, clip): model-file size, load+prefill time, generation time,
peak host RSS, peak VRAM, and whether the transcript is byte-identical to the
f32 baseline.

Prefill is isolated by running the same clip twice -- once with `--max-new 1`,
once unbounded -- and differencing. That needs no instrumentation inside the
vendored engine, which must stay unmodified.

Accuracy is judged against the f32 output on the SAME clip, because f32 is the
gated reference. Byte-identity is the strong signal; where it fails, the
character/word agreement says how far off it is, since a single flipped
near-tied timestamp is not the same defect as a lost sentence.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import threading
import time
from pathlib import Path

BIN = Path("/home/luigi/rs-pure/build-cuda/rs-moss-td")


def vram_now() -> int:
    """Total MiB used across GPUs, ONE sample."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def peak_vram_sampler(stop: threading.Event, out: list) -> None:
    # Loops until `stop` is set. Calling this with a fresh Event to grab a single
    # baseline sample spins forever -- that bug silently burned 25 minutes of
    # "profiling" that measured nothing. Use vram_now() for one-shot samples.
    while not stop.is_set():
        out.append(vram_now())
        stop.wait(0.25)


def run(gguf: Path, wav: Path, max_new: int | None):
    cmd = [str(BIN), "transcribe", str(gguf), str(wav)]
    if max_new is not None:
        cmd += ["--max-new", str(max_new)]
    stop, samples = threading.Event(), []
    t = threading.Thread(target=peak_vram_sampler, args=(stop, samples), daemon=True)
    base = [vram_now()]
    t.start()
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-f", "%M"] + cmd,
                       capture_output=True, text=True)
    wall = time.time() - t0
    stop.set(); t.join(timeout=2)
    rss_kb = 0
    for line in p.stderr.strip().splitlines()[::-1]:
        if line.strip().isdigit():
            rss_kb = int(line.strip()); break
    vram = max(0, (max(samples) if samples else 0) - base[0])
    return {"wall": wall, "rss_gb": rss_kb / 1e6, "vram_mb": vram,
            "text": p.stdout.rstrip("\n")}


def agreement(a: str, b: str) -> float:
    ca, cb = list(a), list(b)
    sm = difflib.SequenceMatcher(None, ca, cb)
    same = sum(bl.size for bl in sm.get_matching_blocks())
    return 100.0 * same / max(1, max(len(ca), len(cb)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--baseline", required=True, help="f32 gguf (the reference)")
    ap.add_argument("--out", default="/tmp/claude-1001/golden/profile_cuda.json")
    args = ap.parse_args()

    results = []
    for clip in args.clips:
        wav = Path(clip)
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout)
        print(f"\n=== {wav.name}  ({dur:.0f}s) ===")
        print(f"{'model':<12} {'size':>6} {'prefill':>8} {'gen':>8} {'total':>8} "
              f"{'RTF':>6} {'RSS':>7} {'VRAM':>8}  accuracy vs f32")
        base_text = None
        for m in args.models:
            g = Path(m)
            pre = run(g, wav, max_new=1)
            full = run(g, wav, max_new=16384)
            gen = max(0.0, full["wall"] - pre["wall"])
            if g.name == Path(args.baseline).name:
                base_text = full["text"]
            acc = ("baseline" if base_text is None or full["text"] == base_text
                   else f"{agreement(base_text, full['text']):.2f}% chars")
            if base_text is not None and full["text"] == base_text and \
               g.name != Path(args.baseline).name:
                acc = "BYTE-IDENTICAL"
            print(f"{g.stem.replace('moss-transcribe-',''):<12} "
                  f"{g.stat().st_size/1e9:>5.2f}G {pre['wall']:>7.1f}s "
                  f"{gen:>7.1f}s {full['wall']:>7.1f}s "
                  f"{full['wall']/dur:>6.3f} {full['rss_gb']:>6.2f}G "
                  f"{full['vram_mb']:>6d}MB  {acc}")
            results.append({"clip": wav.name, "dur": dur, "model": g.name,
                            "size_gb": g.stat().st_size / 1e9,
                            "prefill_s": pre["wall"], "gen_s": gen,
                            "total_s": full["wall"], "rtf": full["wall"] / dur,
                            "rss_gb": full["rss_gb"], "vram_mb": full["vram_mb"],
                            "accuracy": acc})
            Path(f"/tmp/claude-1001/golden/prof_{wav.stem}_{g.stem}.txt").write_text(
                full["text"], encoding="utf-8")
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
