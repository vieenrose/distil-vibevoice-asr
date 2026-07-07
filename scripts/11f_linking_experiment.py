#!/usr/bin/env python
"""Cross-window speaker-linking experiment.

Assumes per-window diarization is good (teacher's local speaker ids). Compares
methods for LINKING those local ids into global identities, on a CLEAN
(zero-overlap) recurring-speaker meeting so segment quality is not the confound.

Methods:
  A per-segment global AHC        (current 'recluster')
  B pooled (window,speaker) AHC
  C pooled + overlap time-anchoring (must-link) constrained AHC
  D pooled spectral, eigengap-k
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.data.pseudo_label import TeacherLabeler
from distil_vibevoice.data.simulate_meetings import simulate_meeting
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.runtime.embeddings import load_embedder

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
WIN, HOP = 90.0, 60.0          # 30 s overlap for anchoring
N_SPK, MEET_MIN = 5, 14.0


def build_clean_meeting(rng):
    from math import gcd

    from scipy.signal import resample_poly
    tsv = next(iter(glob.glob(str(ROOT / "data/raw/common_voice_zhtw/**/test.tsv"), recursive=True)), None)
    mp3 = {os.path.basename(p): p for p in glob.glob(str(ROOT / "data/raw/common_voice_zhtw/**/*.mp3"), recursive=True)}
    import csv
    by: dict[str, list[str]] = {}
    with open(tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n = os.path.basename(r.get("path", ""))
            if n in mp3:
                by.setdefault(r["client_id"], []).append(mp3[n])
    clients = [c for c in sorted(by, key=lambda c: -len(by[c])) if len(by[c]) >= 10][:N_SPK]
    utts, pos, total = [], {c: 0 for c in clients}, 0.0
    while total < MEET_MIN * 60:
        for si, c in enumerate(clients):
            if pos[c] >= len(by[c]):
                pos[c] = 0
            w, sr = sf.read(by[c][pos[c]]); pos[c] += 1
            w = np.asarray(w)
            if w.ndim > 1:
                w = w.mean(1)
            if sr != SR:
                g = gcd(sr, SR); w = resample_poly(w, SR // g, sr // g)
            w = w.astype(np.float32)
            if w.size < SR // 2:
                continue
            utts.append((w, str(si), "")); total += w.size / SR
        if total >= MEET_MIN * 60:
            break
    return simulate_meeting(utts, SR, overlap_ratio=0.0, rng=rng)  # CLEAN, no overlap


def cons(ref, segs, labels):
    hyp = [Segment(s.start, s.end, f"C{l}", "") for s, l in zip(segs, labels)]
    return round(speaker_consistency(ref, hyp), 3), len(set(labels))


def main() -> int:
    rng = np.random.default_rng(3)
    wav, ref = build_clean_meeting(rng)
    wav = np.asarray(wav, dtype=np.float32); dur = len(wav) / SR
    print(f"clean meeting: {dur/60:.1f} min, {len(set(s.speaker for s in ref))} speakers, {len(ref)} turns")
    emb = load_embedder("ecapa")
    lab = TeacherLabeler(str(ROOT / "models/teacher"))

    # transcribe per window, keep (window, local speaker)
    segs, seg_unit, units = [], [], {}   # units[(wi,spk)] -> list of audio chunks
    wi, start = 0, 0.0
    tmp = "/tmp/lw.wav"
    win_span = []
    while start < dur:
        end = min(start + WIN, dur)
        sf.write(tmp, wav[int(start * SR):int(end * SR)], SR)
        rec = lab.label_file(tmp)
        win_span.append((start, end))
        for s in rec.segments:
            if s.text.startswith("["):
                continue
            g0, g1 = start + s.start, start + s.end
            a = wav[int(g0 * SR):int(g1 * SR)]
            if len(a) < int(0.3 * SR):
                continue
            segs.append(Segment(g0, g1, s.speaker, s.text))
            key = (wi, s.speaker)
            seg_unit.append(key)
            units.setdefault(key, []).append(a)
        wi += 1; start += HOP

    # dedupe segments for scoring: keep those whose midpoint is in the window's core
    def core(i):
        a, b = win_span[i]
        return (a if i == 0 else a + (WIN - HOP) / 2, b if b >= dur else b - (WIN - HOP) / 2)
    keep = [j for j, (s, k) in enumerate(zip(segs, seg_unit))
            if core(k[0])[0] <= (s.start + s.end) / 2 < core(k[0])[1]]
    ksegs = [segs[j] for j in keep]; kunit = [seg_unit[j] for j in keep]
    print(f"{len(segs)} teacher segs ({len(ksegs)} after dedupe), {len(units)} (window,speaker) units, {wi} windows")

    # pooled unit embeddings
    ukeys = list(units)
    U = np.stack([emb.embed(np.concatenate(units[k]), SR) for k in ukeys])
    uidx = {k: i for i, k in enumerate(ukeys)}

    # ---- A: per-segment global AHC (on kept segments) ----
    Sm = np.stack([emb.embed(wav[int(s.start * SR):int(s.end * SR)], SR) for s in ksegs])
    Za = linkage(pdist(Sm, "cosine"), "average")
    bestA = max((cons(ref, ksegs, fcluster(Za, t, "distance")) for t in np.arange(0.4, 0.95, 0.05)), key=lambda x: x[0])
    print(f"A per-segment AHC (best thr)      : consistency={bestA[0]} ({bestA[1]} clusters)")

    # map unit-cluster labelings onto kept segments
    def unit_labels_to_segs(unit_lab):
        u2l = {k: unit_lab[uidx[k]] for k in ukeys}
        return [u2l[k] for k in kunit]

    # ---- B: pooled unit AHC ----
    Zu = linkage(pdist(U, "cosine"), "average")
    bestB = max((cons(ref, ksegs, unit_labels_to_segs(fcluster(Zu, t, "distance"))) for t in np.arange(0.4, 0.95, 0.05)), key=lambda x: x[0])
    print(f"B pooled-unit AHC (best thr)      : consistency={bestB[0]} ({bestB[1]} clusters)")

    # ---- C: pooled + overlap time-anchoring must-links -> constrained AHC ----
    # must-link units in adjacent windows whose segments overlap in the shared region
    parent = list(range(len(ukeys)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    unit_segs: dict = {}
    for s, k in zip(segs, seg_unit):
        unit_segs.setdefault(k, []).append(s)
    for i in range(wi - 1):
        ov0, ov1 = win_span[i + 1][0], win_span[i][1]
        for ka in [k for k in ukeys if k[0] == i]:
            for kb in [k for k in ukeys if k[0] == i + 1]:
                # temporal intersection of their segments inside the overlap region
                inter = 0.0
                for sa in unit_segs.get(ka, []):
                    for sb in unit_segs.get(kb, []):
                        lo = max(sa.start, sb.start, ov0); hi = min(sa.end, sb.end, ov1)
                        inter += max(0.0, hi - lo)
                if inter > 1.0:  # >1s co-timed in overlap -> same speaker
                    parent[find(uidx[ka])] = find(uidx[kb])
    # merge must-linked units, average embeddings
    groups: dict = {}
    for k in ukeys:
        groups.setdefault(find(uidx[k]), []).append(uidx[k])
    gkeys = list(groups)
    G = np.stack([U[groups[g]].mean(0) / (np.linalg.norm(U[groups[g]].mean(0)) + 1e-9) for g in gkeys])
    Zg = linkage(pdist(G, "cosine"), "average") if len(G) > 1 else None
    def C_labels(t):
        if Zg is None:
            gl = np.zeros(len(gkeys), dtype=int)
        else:
            gl = fcluster(Zg, t, "distance")
        u2g = {}
        for gi, g in enumerate(gkeys):
            for ui in groups[g]:
                u2g[ukeys[ui]] = gl[gi]
        return [u2g[k] for k in kunit]
    bestC = max((cons(ref, ksegs, C_labels(t)) for t in np.arange(0.4, 0.95, 0.05)), key=lambda x: x[0])
    print(f"C pooled + overlap-anchor (best)  : consistency={bestC[0]} ({bestC[1]} clusters)  [{len(gkeys)} anchored groups]")

    # ---- D: pooled spectral, eigengap-k ----
    aff = 1 - squareform(pdist(U, "cosine")); np.fill_diagonal(aff, 1.0)
    d = np.maximum(aff.sum(1), 1e-9); dinv = 1 / np.sqrt(d)
    L = np.eye(len(aff)) - dinv[:, None] * aff * dinv[None, :]
    ev = np.sort(np.linalg.eigvalsh(L)); k = int(np.argmax(np.diff(ev[:min(10, len(ev))])[1:]) + 2)
    from scipy.cluster.hierarchy import fcluster as fc
    dl = fc(Zu, k, "maxclust")
    consD = cons(ref, ksegs, unit_labels_to_segs(dl))
    print(f"D pooled spectral-k (est k={k})     : consistency={consD[0]} ({consD[1]} clusters)")

    print(f"\ntrue speakers = {N_SPK} | best method: " +
          max([("A", bestA), ("B", bestB), ("C", bestC), ("D", consD)], key=lambda x: x[1][0])[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
