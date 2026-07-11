#!/usr/bin/env python
"""Recover cross-window diarization consistency at a SAFE window (180s).

The 180s window over-segments (123-min meeting: 38 speaker labels found vs 26
ref -> consistency 0.873 vs 300s's 0.905). Over-segmentation = one real speaker
split across windows, a LINKING problem. This sweeps linking strategies to
merge better without collapsing distinct speakers:

  method:   seg  = per-segment AHC (current)
            cent = per-(window, MOSS-speaker) duration-weighted centroid AHC
  linkage:  average / complete / ward
  threshold grid
  merge2:   optional 2nd-pass merge of resulting cluster centroids @ m2 thr

Scores consistency + DER vs the catalog pyannote reference; target = beat the
current 0.873 toward 300s's 0.905 while keeping DER low.
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

ROOT = Path(__file__).resolve().parents[1]


def load_c01b():
    spec = importlib.util.spec_from_file_location(
        "c01b", str(ROOT / "scripts/01b_collect_ivod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ahc(feats, t, linkage_method):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    if len(feats) == 1:
        return np.array([1])
    metric = "euclidean" if linkage_method == "ward" else "cosine"
    Z = linkage(pdist(feats, metric), linkage_method)
    return fcluster(Z, t=t, criterion="distance")


def canon(raw):
    m, out = {}, []
    for x in raw:
        x = int(x)
        if x not in m:
            m[x] = f"S{len(m)+1:02d}"
        out.append(m[x])
    return out


def merge_centroids(labels, feats, m2):
    """2nd pass: merge final clusters whose mean embeddings are within m2."""
    uniq = sorted(set(labels))
    cents = []
    for u in uniq:
        v = np.mean([feats[i] for i in range(len(labels)) if labels[i] == u], 0)
        n = np.linalg.norm(v)
        cents.append(v / n if n > 1e-8 else v)
    merged = canon(ahc(np.stack(cents), m2, "average"))
    remap = {u: merged[i] for i, u in enumerate(uniq)}
    return [remap[l] for l in labels]


def link(segs, embs, method, thr, linkage_method, m2, min_core=3.0):
    idx = [i for i in range(len(segs)) if i in embs
           and (method == "seg" and segs[i]["end"] - segs[i]["start"] >= min_core
                or method == "cent")]
    if method == "seg":
        pts_idx = idx
        feats = [embs[i] for i in idx]
        if len(feats) < 2:
            return None
        raw = canon(ahc(np.stack(feats), thr, linkage_method))
        if m2:
            raw = merge_centroids(raw, feats, m2)
        lab_of = dict(zip(pts_idx, raw))
        # assign non-core to nearest cluster centroid
        cent = {}
        for i, l in lab_of.items():
            cent.setdefault(l, []).append(embs[i])
        keys = list(cent)
        M = np.stack([np.mean(cent[k], 0) for k in keys])
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        out, prev = [], "S01"
        for i, s in enumerate(segs):
            if i in lab_of:
                spk = lab_of[i]
            elif i in embs:
                spk = keys[int(np.argmax(M @ embs[i]))]
            else:
                spk = prev
            prev = spk
            out.append(Segment(s["start"], s["end"], spk, s["text"]))
        return out
    # cent: cluster per-(window, speaker) duration-weighted centroids
    groups = {}
    for i, s in enumerate(segs):
        if i in embs:
            groups.setdefault((s["win"], s["win_speaker"]), []).append(i)
    keys, cents = [], []
    for k, idxs in groups.items():
        w = np.array([max(0.1, segs[i]["end"] - segs[i]["start"]) for i in idxs])
        c = np.average(np.stack([embs[i] for i in idxs]), 0, weights=w)
        n = np.linalg.norm(c)
        if n > 1e-8:
            keys.append(k)
            cents.append(c / n)
    if len(cents) < 2:
        return None
    raw = canon(ahc(np.stack(cents), thr, linkage_method))
    if m2:
        raw = merge_centroids(raw, cents, m2)
    lab_of = dict(zip(keys, raw))
    out, prev = [], "S01"
    for s in segs:
        spk = lab_of.get((s["win"], s["win_speaker"]), prev)
        prev = spk
        out.append(Segment(s["start"], s["end"], spk, s["text"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/chunk_dump_v4win")
    ap.add_argument("--window", default="180")
    ap.add_argument("--stems", nargs="+",
                    default=["ivod_2024_15857", "ivod_2024_15361"])
    args = ap.parse_args()
    c01b = load_c01b()

    data = {}
    for stem in args.stems:
        d = json.loads((ROOT / args.dump / f"{stem}.json").read_text())
        embs_all = dict(np.load(ROOT / args.dump / f"{stem}.npz"))
        embs = {int(k.split("/", 1)[1]): v for k, v in embs_all.items()
                if k.startswith(args.window + "/")}
        tr = d.get("meta_ref", {}).get("transcript") or {}
        wx = tr.get("whisperx") or []
        skip = c01b.robust_speech_start(wx) if wx else 0.0
        ref = [Segment(p["start"]-skip, p["end"]-skip, str(p["speaker"]), "")
               for p in (tr.get("pyannote") or [])
               if p.get("end", 0)-skip > 0 and p.get("start", 1e12)-skip < d["duration"]]
        data[stem] = (d["chunked"][args.window], embs, ref, d["duration"])

    grid = []
    for method in ("seg", "cent"):
        for lk in ("average", "complete", "ward"):
            for thr in (0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 1.10):
                for m2 in (0.0, 0.5, 0.7):
                    grid.append((method, lk, thr, m2))

    print(f"window={args.window}s | current baseline: 15857 cons 0.873 / "
          f"300s 0.905\n")
    results = []
    for method, lk, thr, m2 in grid:
        row = {}
        ok = True
        for stem, (segs, embs, ref, dur) in data.items():
            hyp = link(segs, embs, method, thr, lk, m2)
            if hyp is None or not ref:
                ok = False
                break
            row[stem] = (speaker_consistency(ref, hyp), der(ref, hyp),
                         len(set(s.speaker for s in hyp)))
        if ok:
            # rank by consistency on the long meeting, DER as tiebreak
            long = row.get("ivod_2024_15857")
            results.append((long[0], -long[1], method, lk, thr, m2, row))

    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    print("TOP configs (by 123-min consistency):")
    for cons, negder, method, lk, thr, m2, row in results[:12]:
        s = " | ".join(f"{k[-5:]}: cons={v[0]:.3f} DER={v[1]:.3f} n={v[2]}"
                       for k, v in row.items())
        print(f"  {method:4s} {lk:8s} t={thr:.2f} m2={m2:.1f} :: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
