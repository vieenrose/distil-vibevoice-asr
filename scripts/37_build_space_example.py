#!/usr/bin/env python
"""Build the HF Space precomputed-example JSON from a chunk dump.

Takes the winning linking config from 34b_sweep_linking and writes
space/examples/<stem>.json in the format app.py's load_cached() expects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from distil_vibevoice.runtime.linking import (AHC_THRESHOLD, MIN_CORE_DUR_S,
                                              link_speakers)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="ivod_2024_15857")
    ap.add_argument("--window", default="300")
    args = ap.parse_args()

    d = json.loads((ROOT / f"data/chunk_dump/{args.stem}.json").read_text(
        encoding="utf-8"))
    embs_all = dict(np.load(ROOT / f"data/chunk_dump/{args.stem}.npz"))
    segs = d["chunked"][args.window]
    embs = {int(k.split("/", 1)[1]): v for k, v in embs_all.items()
            if k.startswith(args.window + "/")}
    labels = link_speakers(segs, embs)

    out = {
        "audio": f"{args.stem}.mp3",
        "pipeline": {"window_s": int(args.window),
                     "method": "core-seg AHC + nearest-centroid",
                     "threshold": AHC_THRESHOLD,
                     "min_core_dur_s": MIN_CORE_DUR_S,
                     "model": "Luigi/moss-transcribe-diarize-zhtw"},
        "segments": [{"start": s["start"], "end": s["end"],
                      "speaker": lab, "text": s["text"]}
                     for s, lab in zip(segs, labels)],
    }
    dst = ROOT / f"space/examples/{args.stem}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{dst}: {len(segs)} segments, {len(set(labels))} speakers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
