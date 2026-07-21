#!/usr/bin/env python3
"""Windowed pipeline over the vendored C API, swept across window lengths.

This is a WRAPPER, not an engine change: it calls
moss_transcribe_capi_transcribe_pcm() once per window and stitches results in
Python. moss_td/ stays byte-for-byte upstream.

Windowing itself is deliberately minimal -- pause-aware cut near the window
boundary, keep segments before the cut, advance the cursor -- and nothing else.
The old WASM/C++ demo accumulated many more harnesses on top of this (ragged-
tail retreat, marker-less fallback, EOS-coverage suppression, loop guards) after
they were each added to patch a specific observed failure; none of those are
reproduced here so the window-length search measures windowing's own effect,
not a pile of compensating hacks.

Usage:
  .venv/bin/python scripts/85_window_sweep.py AUDIO.wav --windows 90 180 300 450 \\
      --gguf q8mix.gguf --out-prefix /tmp/.../win
"""
from __future__ import annotations

import argparse
import ctypes
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?\](?:\[(S\d+)\])?([^\[]*)")


def load_engine(so_path: str, gguf: str):
    lib = ctypes.CDLL(so_path)
    lib.moss_transcribe_capi_load.restype = ctypes.c_void_p
    lib.moss_transcribe_capi_load.argtypes = [ctypes.c_char_p]
    lib.moss_transcribe_capi_transcribe_pcm.restype = ctypes.c_void_p
    lib.moss_transcribe_capi_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.c_int]
    lib.moss_transcribe_capi_free_string.argtypes = [ctypes.c_void_p]
    ctx = lib.moss_transcribe_capi_load(gguf.encode())
    assert ctx, "model load failed"
    return lib, ctx


def transcribe(lib, ctx, pcm: np.ndarray, max_new: int) -> str:
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    p = lib.moss_transcribe_capi_transcribe_pcm(
        ctypes.c_void_p(ctx), pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(pcm), SR, max_new)
    if not p:
        return ""
    try:
        return ctypes.cast(p, ctypes.c_char_p).value.decode("utf-8", "replace")
    finally:
        lib.moss_transcribe_capi_free_string(p)


def pause_cut(piece: np.ndarray, window_s: float, snap_s: float) -> float:
    """Seconds into `piece` of the quietest 0.4s frame within the last snap_s of
    the window -- so a cut lands in silence, not mid-utterance. Mirrors the old
    JS pauseCut exactly (same window/hop/snap constants)."""
    n = len(piece)
    if n < window_s * SR:
        return n / SR
    frm = max(0, n - int(snap_s * SR))
    win, hop = int(0.4 * SR), int(0.1 * SR)
    best, best_e = n - win, float("inf")
    o = frm
    while o + win <= n:
        e = float(np.sum(piece[o:o + win] ** 2))
        if e < best_e:
            best_e, best = e, o
        o += hop
    return (best + win / 2) / SR


def parse_window(text: str, win_start_s: float):
    segs, prev_spk = [], "S01"
    for m in SEG_RE.finditer(text):
        t = float(m.group(1))
        if m.group(3):
            prev_spk = m.group(3)
        body = (m.group(4) or "").strip()
        if body:
            segs.append({"start": win_start_s + t,
                        "rawEnd": win_start_s + float(m.group(2)) if m.group(2) else None,
                        "spk": prev_spk, "text": body})
    return segs


def run_windowed(lib, ctx, pcm: np.ndarray, window_s: float, max_new: int):
    dur_s = len(pcm) / SR
    snap_s = 12 if window_s >= 90 else 5
    cursor, all_segs = 0.0, []
    win_idx = 0
    while cursor < dur_s - 0.5:
        frm = int(cursor * SR)
        piece = pcm[frm: frm + int(window_s * SR)]
        if len(piece) < SR:
            break
        is_last = frm + len(piece) >= len(pcm)
        cut = pause_cut(piece, window_s, snap_s) if not is_last else len(piece) / SR
        win_start = cursor
        text = transcribe(lib, ctx, piece[:int(cut * SR)], max_new)
        segs = parse_window(text, win_start)
        cut_abs = win_start + cut
        kept = [s for s in segs if s["start"] < cut_abs - 0.01]
        for s in kept:
            s["win"] = win_idx           # unit key for pooled speaker linking
            s["local_spk"] = s["spk"]
        all_segs.extend(kept)
        cursor = cut_abs
        win_idx += 1
        if is_last:
            break
    return all_segs


def render(segs):
    out = []
    for i, s in enumerate(segs):
        end = segs[i + 1]["start"] if i + 1 < len(segs) else s.get("rawEnd")
        out.append(f"[{s['start']:.2f}][{s['spk']}]{s['text']}")
    return "".join(out) + (f"[{segs[-1].get('rawEnd') or segs[-1]['start']:.2f}]" if segs else "")


# --------------------------------------------------------- speaker linking ---
# Each window's [Sxx] tags are a LOCAL numbering that the model resets every
# call -- there is no engine-side notion of "the same voice as window 3". Measured
# effect on the AMI clip: word accuracy is untouched (WER 0.15-0.16 windowed vs
# 0.159 windowless) but speaker accuracy collapses 99.4% -> ~50%, because the
# label assigned to a given voice is arbitrary per window.
#
# Fix: CAM++ (rapidspeech-core's rs_speaker_* C API, unmodified) embeds each
# utterance's audio; utterances are greedily assigned to the most similar
# EXISTING global-speaker centroid (cosine >= threshold) or start a new one.
# This is presentation-layer linking bolted onto the raw per-window output, not
# an engine change -- exactly the same category of wrapper as the windowing
# loop itself.
def load_speaker_model(so_path: str, gguf: str):
    lib = ctypes.CDLL(so_path)
    lib.rs_speaker_init_from_file.restype = ctypes.c_void_p
    lib.rs_speaker_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.rs_speaker_dim.argtypes = [ctypes.c_void_p]
    lib.rs_speaker_dim.restype = ctypes.c_int32
    lib.rs_speaker_embed.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    sp = lib.rs_speaker_init_from_file(gguf.encode(), 0)
    assert sp, "campplus load failed"
    dim = lib.rs_speaker_dim(ctypes.c_void_p(sp))
    return lib, sp, dim


def embed(lib, sp, dim, pcm: np.ndarray):
    if len(pcm) < 1600:            # <0.1s: too short for a stable embedding
        return None
    buf = (ctypes.c_float * dim)()
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    err = lib.rs_speaker_embed(ctypes.c_void_p(sp),
                               pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               len(pcm), buf, dim)
    if err != 0:
        return None
    v = np.frombuffer(buf, dtype=np.float32).copy()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def link_speakers(lib, sp, dim, pcm: np.ndarray, segs: list, threshold: float = 0.55):
    """Link at the (window, local-tag) UNIT level, not per-utterance.

    First attempt embedded each utterance alone and it fragmented badly (up to
    117 "speakers" on a 4-speaker meeting): 39% of utterances are under 2s
    (single-word backchannels like "Yeah." / "Okay."), too little audio for a
    stable CAM++ embedding, so same-speaker cosine similarity was noisy enough
    to straddle any single threshold. Pooling all of one window-local speaker's
    audio into one embedding gives CAM++ 10-90s to work with instead of ~1-3s,
    which is the "unit pooling" fix already used successfully elsewhere in this
    project's speaker-linking history.
    """
    dur_s = len(pcm) / SR
    units: dict[tuple, list] = {}
    for i, s in enumerate(segs):
        key = (s["win"], s["local_spk"])
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else dur_s
        end = min(nxt, s["start"] + 12.0, dur_s)
        units.setdefault(key, []).append((s["start"], end))

    unit_emb = {}
    for key, ranges in units.items():
        # Pool up to ~30s of this unit's audio (evenly sampled across its
        # ranges, not just the first few seconds) into one embedding call.
        total = sum(e - a for a, e in ranges)
        budget, chunks = 30.0, []
        for a, e in sorted(ranges):
            take = min(e - a, max(0.0, budget))
            if take <= 0:
                break
            chunks.append(pcm[int(a * SR):int((a + take) * SR)])
            budget -= take
        pooled = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        unit_emb[key] = (embed(lib, sp, dim, pooled), total)

    # CONSTRAINED AGGLOMERATIVE clustering, not greedy time-order streaming.
    #
    # Greedy streaming (process units in time order, attach each to the best
    # existing centroid or start a new one) has no error correction: one bad
    # merge permanently contaminates a centroid via its running mean, and a
    # contaminated centroid is EASIER to falsely match against later units of
    # either real speaker, so the error compounds. Measured on the 300s window:
    # this merged two individually pure (0.94-1.00) real speakers into one
    # identity at 57%/43% purity, corrupting 1777 matched words.
    #
    # Fix has two parts:
    #  1. Merge globally-best-first (standard AHC), not in time order, so one
    #     ambiguous early pair can't seed a bad cluster before better evidence
    #     arrives.
    #  2. CANNOT-LINK constraints: two units from the SAME window with
    #     DIFFERENT local tags are provably different real speakers (raw
    #     per-window tag purity measured 0.94-1.00 -- the engine's own within-
    #     window diarization is reliable). No merge may combine two clusters
    #     that contain a cannot-link pair, even transitively. This makes the
    #     exact 300s failure (merging a real B-cluster with a real D-cluster)
    #     structurally impossible rather than just unlikely.
    keys = [k for k in units if unit_emb[k][0] is not None]
    orphans = [k for k in units if unit_emb[k][0] is None]
    n = len(keys)
    clusters = [{k} for k in keys]                       # cluster i = clusters[i]
    cluster_emb = {i: unit_emb[keys[i]][0].copy() for i in range(n)}
    cluster_n = {i: 1 for i in range(n)}
    cannot_link = set()
    by_win: dict = {}
    for k in keys:
        by_win.setdefault(k[0], []).append(k)
    for win, ks in by_win.items():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                cannot_link.add(frozenset((ks[a], ks[b])))

    # Cannot-link is a strong PRIOR, not an inviolable rule. The engine mostly
    # keeps one tag per speaker per window (6 of 7 windows on the AMI clip are
    # clean), but it does sometimes over-split: window 2 gave real speaker B two
    # tags (S02, S03). Treating that as proof-of-distinctness made the max
    # local-tag count in ANY window a hard floor on the global speaker count --
    # 5 tags in window 2 meant >=5 clusters no matter how permissive the
    # threshold, which is why lowering it plateaued at 6-7 instead of reaching 4.
    # So: require a much higher similarity to merge across a same-window pair,
    # rather than forbidding it. Overwhelming acoustic evidence can override the
    # engine's split; ordinary evidence cannot.
    CONSTRAINT_PENALTY = 0.35
    alive = set(range(n))
    while True:
        best_pair, best_sim, best_eff = None, None, threshold
        alive_l = list(alive)
        for ai in range(len(alive_l)):
            for bi in range(ai + 1, len(alive_l)):
                i, j = alive_l[ai], alive_l[bi]
                constrained = any(frozenset((u, v)) in cannot_link
                                  for u in clusters[i] for v in clusters[j])
                sim = float(np.dot(cluster_emb[i], cluster_emb[j]))
                eff = sim - (CONSTRAINT_PENALTY if constrained else 0.0)
                if eff > best_eff:
                    best_eff, best_sim, best_pair = eff, sim, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] |= clusters[j]
        cluster_emb[i] = (cluster_emb[i] * cluster_n[i] + cluster_emb[j] * cluster_n[j])
        cluster_emb[i] /= np.linalg.norm(cluster_emb[i])
        cluster_n[i] += cluster_n[j]
        alive.discard(j)

    # Absorb TINY clusters into their best non-tiny match.
    #
    # After constrained AHC the big clusters are right (measured on AMI: 4
    # clusters of 1019/774/362/235 words mapping exactly onto the 4 real
    # speakers), but a couple of 3-12 word fragments survive as their own
    # "speakers". They are the same failure as the original per-utterance
    # attempt: too little audio for a stable CAM++ embedding, so their vector
    # is noise and no threshold places them correctly. Tuning the threshold
    # or the cannot-link penalty does NOT fix this -- measured identical
    # results across penalty 0.25/0.35 x threshold 0.45/0.50.
    #
    # Rather than let noise-embeddings define speakers, attach each tiny
    # cluster to whichever substantial cluster it is closest to (ignoring the
    # merge threshold, since we have already decided it is not its own
    # speaker), and let cannot-link still veto the impossible ones.
    MIN_UNIT_SECONDS = 8.0
    cluster_dur = {i: sum(unit_emb[k][1] for k in clusters[i]) for i in alive}
    big = {i for i in alive if cluster_dur[i] >= MIN_UNIT_SECONDS}
    tiny = sorted(alive - big, key=lambda i: cluster_dur[i])
    for i in tiny:
        if not big:
            break
        cands = []
        for j in big:
            if any(frozenset((u, v)) in cannot_link
                   for u in clusters[i] for v in clusters[j]):
                continue
            cands.append((float(np.dot(cluster_emb[i], cluster_emb[j])), j))
        if not cands:
            continue
        _, j = max(cands)
        clusters[j] |= clusters[i]
        alive.discard(i)

    unit_gid = {}
    for gid, i in enumerate(sorted(alive)):
        for k in clusters[i]:
            unit_gid[k] = gid
    for k in orphans:
        unit_gid[k] = None

    for s in segs:
        gid = unit_gid.get((s["win"], s["local_spk"]))
        s["spk"] = f"S{gid + 1:02d}" if gid is not None else s["local_spk"]
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("--windows", nargs="+", type=float, required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--so", default="/home/luigi/rs-pure/build-cuda/libmoss_td.so")
    ap.add_argument("--tokens-per-audio-second", type=float, default=12.0)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--speaker-gguf", default=None,
                    help="campplus.gguf; if given, links speaker identity across "
                         "windows instead of using each window's local [Sxx]")
    ap.add_argument("--core-so",
                    default="/home/luigi/rs-pure/build-cuda/librapidspeech-core.so")
    ap.add_argument("--link-threshold", type=float, default=0.65)
    args = ap.parse_args()

    pcm, sr = sf.read(args.audio, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr}"

    lib, ctx = load_engine(args.so, args.gguf)
    splib = sp = dim = None
    if args.speaker_gguf:
        splib, sp, dim = load_speaker_model(args.core_so, args.speaker_gguf)

    for w in args.windows:
        max_new = max(5120, int(args.tokens_per_audio_second * w))
        segs = run_windowed(lib, ctx, pcm, w, max_new)
        if splib is not None:
            segs = link_speakers(splib, sp, dim, pcm, segs, args.link_threshold)
        text = render(segs)
        out = f"{args.out_prefix}_w{int(w)}.txt"
        Path(out).write_text(text, encoding="utf-8")
        n_spk = len({s["spk"] for s in segs})
        print(f"window={w:.0f}s -> {len(segs)} segments, {n_spk} speakers, "
              f"{len(text)} bytes -> {out}")


if __name__ == "__main__":
    main()
