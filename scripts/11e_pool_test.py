#!/usr/bin/env python
"""Per-window-speaker embedding pooling test: does it close the 0.86->0.95 gap?

Within one window the teacher's local speaker ids are reliable, so pool ALL of a
local speaker's audio into ONE clean ECAPA embedding (instead of many noisy
short-segment ones), then cluster the pooled embeddings across windows.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.data.pseudo_label import TeacherLabeler
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.runtime.embeddings import load_embedder

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
WIN, HOP = 120.0, 100.0  # 20s overlap


def main() -> int:
    spec = importlib.util.spec_from_file_location("s11c", str(ROOT / "scripts/11c_eval_synthetic.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _, ref = m.build_long_recurring(5, 15.0, np.random.default_rng(7))
    wav, _ = sf.read(str(ROOT / "data/eval_synth/long_meeting.wav"))
    wav = np.asarray(wav, dtype=np.float32)
    dur = len(wav) / SR

    emb = load_embedder("ecapa")
    lab = TeacherLabeler(str(ROOT / "models/teacher"))

    all_segs: list[Segment] = []          # global segments (for scoring)
    pooled: list[np.ndarray] = []         # one embedding per (window, local spk)
    pooled_key: list[tuple[int, str]] = []
    seg_key: list[tuple[int, str]] = []   # (window, local spk) per global segment

    wi, start = 0, 0.0
    tmp = "/tmp/pool_win.wav"
    while start < dur:
        end = min(start + WIN, dur)
        sf.write(tmp, wav[int(start * SR):int(end * SR)], SR)
        rec = lab.label_file(tmp)
        by_spk: dict[str, list[np.ndarray]] = {}
        for s in rec.segments:
            if s.text.startswith("["):
                continue
            g0, g1 = start + s.start, start + s.end
            all_segs.append(Segment(g0, g1, s.speaker, s.text))
            seg_key.append((wi, s.speaker))
            a = wav[int(g0 * SR):int(g1 * SR)]
            if len(a) >= int(0.3 * SR):
                by_spk.setdefault(s.speaker, []).append(a)
        for spk, chunks in by_spk.items():
            pooled.append(emb.embed(np.concatenate(chunks), SR))
            pooled_key.append((wi, spk))
        wi += 1
        start += HOP

    P = np.array(pooled)
    print(f"{len(all_segs)} segs, {len(P)} pooled (window,speaker) embeddings, {wi} windows, true=5")
    Z = linkage(pdist(P, "cosine"), "average")
    for thr in (0.6, 0.7, 0.8):
        labels = fcluster(Z, thr, "distance")
        k2c = {k: c for k, c in zip(pooled_key, labels)}
        hyp = [Segment(s.start, s.end, f"C{k2c[k]}", "") for s, k in zip(all_segs, seg_key)]
        print(f"  pooled cluster @{thr}: consistency={speaker_consistency(ref, hyp):.3f} ({len(set(labels))} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
