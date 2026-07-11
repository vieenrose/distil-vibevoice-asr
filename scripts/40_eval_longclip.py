#!/usr/bin/env python
"""Long-clip gate evaluation: compare a model's chunk dumps against the gates.

Reads dumps produced by 36_dump_moss_outputs.py (which stores raw generated
text per window) and reports, per meeting and window size:

  leakage   %% of CJK chars the s2tw conversion changes (raw Simplified output)
  tagdrop   1 - strict-parser segments / lenient-parser segments
  DER/cons  vs the catalog pyannote reference, using the validated linking
            (core-seg AHC @0.45 + nearest-centroid)

Gates (FT v3): leakage < 1%%, tagdrop < 2%%, DER/cons no worse than v2
(0.056/0.891 @30min, 0.180/0.912 @123min, 300 s windows).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.der import der
from distil_vibevoice.runtime.linking import link_speakers

ROOT = Path(__file__).resolve().parents[1]


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/chunk_dump_v3")
    ap.add_argument("--window", default="300")
    args = ap.parse_args()

    from opencc import OpenCC
    cc = OpenCC("s2tw")
    c01b = load_mod("c01b", "scripts/01b_collect_ivod.py")

    import sys
    sys.path.insert(0, str(ROOT / "vendor/MOSS"))
    from moss_transcribe_diarize import parse_transcript as strict_parse

    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    for jpath in sorted((ROOT / args.dump).glob("*.json")):
        d = json.loads(jpath.read_text(encoding="utf-8"))
        embs_all = dict(np.load(jpath.with_suffix(".npz")))
        segs = d["chunked"][args.window]
        raws = (d.get("raw") or {}).get(args.window) or []

        # -- leakage + tag-drop from raw window texts ------------------------
        cjk = changed = 0
        n_strict = n_lenient = 0
        for raw in raws:
            conv = cc.convert(raw)
            for a, b in zip(raw, conv):
                if "一" <= a <= "鿿":
                    cjk += 1
                    if a != b:
                        changed += 1
            n_strict += len(strict_parse(raw))
            n_lenient += len(parse_transcript_lenient(raw))
        leak = 100.0 * changed / max(1, cjk)
        tagdrop = 100.0 * (1 - n_strict / max(1, n_lenient))

        # -- diarization vs pyannote ref --------------------------------------
        embs = {int(k.split("/", 1)[1]): v for k, v in embs_all.items()
                if k.startswith(args.window + "/")}
        labels = link_speakers(segs, embs)
        hyp = [Segment(s["start"], s["end"], lab, s["text"])
               for s, lab in zip(segs, labels)]
        tr = d.get("meta_ref", {}).get("transcript") or {}
        wx = tr.get("whisperx") or []
        skip = c01b.robust_speech_start(wx) if wx else 0.0
        ref = [Segment(p["start"] - skip, p["end"] - skip, str(p["speaker"]), "")
               for p in (tr.get("pyannote") or [])
               if p.get("end", 0) - skip > 0
               and p.get("start", 1e12) - skip < d["duration"]]
        line = (f"{d['stem']:22s} win={args.window}s: "
                f"leak={leak:5.2f}% tagdrop={tagdrop:5.2f}% "
                f"segs={len(segs)} nspk={len(set(labels))}")
        if ref:
            line += (f" cons={speaker_consistency(ref, hyp):.3f} "
                     f"DER={der(ref, hyp):.3f} (ref {len(set(s.speaker for s in ref))} spk)")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
