#!/usr/bin/env python
"""Build the HF Space precomputed-example JSON from a chunk dump.

Takes the winning linking config from 34b_sweep_linking and writes
space/examples/<stem>.json in the format app.py's load_cached() expects.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_sweep():
    spec = importlib.util.spec_from_file_location(
        "s34b", str(ROOT / "scripts/34b_sweep_linking.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="ivod_2024_15857")
    ap.add_argument("--window", default="300")
    ap.add_argument("--method", choices=["seg", "cent"], default="cent")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    s34b = load_sweep()
    d = json.loads((ROOT / f"data/chunk_dump/{args.stem}.json").read_text(
        encoding="utf-8"))
    embs_all = dict(np.load(ROOT / f"data/chunk_dump/{args.stem}.npz"))
    segs = d["chunked"][args.window]
    embs = {k.split("/", 1)[1]: v for k, v in embs_all.items()
            if k.startswith(args.window + "/")}
    fn = s34b.link_seg if args.method == "seg" else s34b.link_cent
    linked = fn(segs, embs, args.threshold)
    assert linked is not None

    out = {
        "audio": f"{args.stem}.mp3",
        "pipeline": {"window_s": int(args.window), "method": args.method,
                     "threshold": args.threshold,
                     "model": "Luigi/moss-transcribe-diarize-zhtw"},
        "segments": [{"start": s.start, "end": s.end,
                      "speaker": s.speaker, "text": s.text} for s in linked],
    }
    dst = ROOT / f"space/examples/{args.stem}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_spk = len({s.speaker for s in linked})
    print(f"{dst}: {len(linked)} segments, {n_spk} speakers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
