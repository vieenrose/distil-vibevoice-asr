#!/usr/bin/env python
"""Sweep cross-window speaker-linking methods on dumped MOSS outputs (no GPU).

Reads data/chunk_dump/<stem>.{json,npz} (from 36_dump_moss_outputs.py) and
scores, against the catalog's pyannote reference:

  method 'seg'  — AHC over per-segment ECAPA embeddings (ignores MOSS's
                  within-window labels)
  method 'cent' — AHC over per-(window, MOSS-speaker) duration-weighted
                  centroids; within-window labels are kept intact and only
                  window-level speakers get merged across windows

across cosine-distance thresholds. Reports consistency + DER + #speakers.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.der import der

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "data/chunk_dump"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "c01b", str(ROOT / "scripts/01b_collect_ivod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ahc(feats: np.ndarray, t: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    if len(feats) == 1:
        return np.array([1])
    return fcluster(linkage(pdist(feats, "cosine"), "average"),
                    t=t, criterion="distance")


def canon_labels(raw: list[int]) -> list[str]:
    canon: dict[int, str] = {}
    out = []
    for lab in raw:
        if lab not in canon:
            canon[lab] = f"S{len(canon) + 1:02d}"
        out.append(canon[lab])
    return out


def link_seg(segs, embs, t):
    idx = [i for i in range(len(segs)) if str(i) in embs]
    if len(idx) < 2:
        return None
    labels = ahc(np.stack([embs[str(i)] for i in idx]), t)
    lab_map = dict(zip(idx, canon_labels([int(x) for x in labels])))
    out, prev = [], "S01"
    for i, s in enumerate(segs):
        spk = lab_map.get(i, prev)
        prev = spk
        out.append(Segment(s["start"], s["end"], spk, s["text"]))
    return out


def link_cent(segs, embs, t):
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(segs):
        groups.setdefault((s["win"], s["win_speaker"]), []).append(i)
    keys, cents = [], []
    for k, idxs in groups.items():
        vecs, wts = [], []
        for i in idxs:
            if str(i) in embs:
                vecs.append(embs[str(i)])
                wts.append(max(0.1, segs[i]["end"] - segs[i]["start"]))
        if not vecs:
            continue
        c = np.average(np.stack(vecs), axis=0, weights=np.asarray(wts))
        n = float(np.linalg.norm(c))
        if n > 1e-8:
            keys.append(k)
            cents.append(c / n)
    if len(cents) < 2:
        return None
    labels = canon_labels([int(x) for x in ahc(np.stack(cents), t)])
    lab_map = dict(zip(keys, labels))
    out, prev = [], "S01"
    for s in segs:
        spk = lab_map.get((s["win"], s["win_speaker"]), prev)
        prev = spk
        out.append(Segment(s["start"], s["end"], spk, s["text"]))
    return out


def main() -> int:
    c01b = load_collector()
    for jpath in sorted(DUMP.glob("*.json")):
        d = json.loads(jpath.read_text(encoding="utf-8"))
        embs_all = dict(np.load(DUMP / f"{d['stem']}.npz"))
        tr = d.get("meta_ref", {}).get("transcript") or {}
        wx = tr.get("whisperx") or []
        skip_s = c01b.robust_speech_start(wx) if wx else 0.0
        py = tr.get("pyannote") or []
        ref = [Segment(p["start"] - skip_s, p["end"] - skip_s,
                       str(p["speaker"]), "")
               for p in py
               if p.get("end", 0) - skip_s > 0
               and p.get("start", 1e12) - skip_s < d["duration"]]
        print(f"\n===== {d['stem']} ({d['duration']/60:.0f} min, "
              f"ref {len(set(s.speaker for s in ref))} spk) =====")
        if not ref:
            print("  no reference; skipping")
            continue

        if d.get("single"):
            hyp = [Segment(s["start"], s["end"], s["speaker"], s["text"])
                   for s in d["single"]]
            print(f"  single-pass          : cons={speaker_consistency(ref, hyp):.3f} "
                  f"DER={der(ref, hyp):.3f} nspk={len(set(s.speaker for s in hyp))}")

        for win_key, segs in d["chunked"].items():
            embs = {k.split("/", 1)[1]: v for k, v in embs_all.items()
                    if k.startswith(win_key + "/")}
            best = {}
            for meth, fn in [("seg", link_seg), ("cent", link_cent)]:
                for t in THRESHOLDS:
                    hyp = fn(segs, embs, t)
                    if hyp is None:
                        continue
                    c = speaker_consistency(ref, hyp)
                    dv = der(ref, hyp)
                    n = len(set(s.speaker for s in hyp))
                    print(f"  win={win_key}s {meth:4s} t={t:.2f}: cons={c:.3f} "
                          f"DER={dv:.3f} nspk={n}")
                    if meth not in best or c > best[meth][1]:
                        best[meth] = (t, c, dv, n)
            for meth, (t, c, dv, n) in best.items():
                print(f"  BEST win={win_key}s {meth}: t={t:.2f} cons={c:.3f} "
                      f"DER={dv:.3f} nspk={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
