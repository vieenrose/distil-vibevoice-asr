#!/usr/bin/env python
"""Validate chunked diarization vs single-pass on REAL meetings (MOSS model).

The phone design processes long meetings in windows and links speakers across
windows via per-segment ECAPA embeddings + global agglomerative reclustering
(the experimentally-chosen method, 0.93 consistency on clean audio). MOSS can
also process the whole meeting in ONE pass on a workstation GPU — giving a
direct measure of what chunking costs on the same real audio:

  1. single-pass MOSS  -> segments (the "no chunking" upper reference)
  2. chunked MOSS      -> per-window [Sxx] labels, offset to global time
  3. link chunked windows: ECAPA embed each segment, global AHC @0.7
  4. score: chunked-vs-singlepass speaker consistency (chunking degradation),
            and both vs the IVOD catalog's pyannote diarization (weak real ref)

Run on GPU: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/34_validate_chunking.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.der import der
from distil_vibevoice.runtime.embeddings import load_embedder

ROOT = Path(__file__).resolve().parents[1]
SR = 24000


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "c01b", str(ROOT / "scripts/01b_collect_ivod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def moss_transcribe(model, proc, wav: np.ndarray, sr: int, dev) -> list[Segment]:
    """Run MOSS on a waveform, return parsed segments."""
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient
    tgt_sr = proc.feature_extractor.sampling_rate
    if sr != tgt_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, tgt_sr)
        wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
    # pass the waveform straight to the processor (process_audio_info chokes
    # on ndarray inputs: `item.get("audio") or ...` is ambiguous for arrays)
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "placeholder.wav"},
        {"type": "text", "text": DEFAULT_PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, audio=[wav], return_tensors="pt").to(dev, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    gen = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [Segment(s.start, s.end, s.speaker, s.text)
            for s in parse_transcript_lenient(gen)]


def link_global(wav: np.ndarray, sr: int, segs: list[Segment],
                embedder, threshold: float = 0.7) -> list[Segment]:
    """Per-segment ECAPA embeddings + global AHC -> global speaker labels."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    feats, keep = [], []
    for s in segs:
        a = wav[int(s.start * sr):int(s.end * sr)]
        if len(a) < int(0.3 * sr):
            continue
        feats.append(embedder.embed(np.asarray(a, dtype=np.float32), sr))
        keep.append(s)
    if len(feats) < 2:
        return segs
    labels = fcluster(linkage(pdist(np.stack(feats), "cosine"), "average"),
                      t=threshold, criterion="distance")
    canon: dict[int, str] = {}
    out = []
    for s, lab in zip(keep, labels):
        lab = int(lab)
        if lab not in canon:
            canon[lab] = f"S{len(canon) + 1:02d}"
        out.append(Segment(s.start, s.end, canon[lab], s.text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v2")
    ap.add_argument("--manifest", default="data/raw/ivod_eval/manifest.jsonl")
    ap.add_argument("--window-s", type=float, default=600.0, help="chunk window (10 min)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    embedder = load_embedder("ecapa")
    c01b = load_collector()

    rows = [json.loads(l) for l in open(ROOT / args.manifest)]
    for rec in rows:
        wav, sr = sf.read(rec["audio_path"])
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), dtype=np.float32)
        dur = len(wav) / sr
        print(f"\n===== {Path(rec['audio_path']).name}: {dur/60:.1f} min =====", flush=True)

        # --- pyannote reference, shifted by the collector's dead-air skip ---
        tr = rec.get("meta", {}).get("transcript") or {}
        wx = tr.get("whisperx") or []
        skip_s = c01b.robust_speech_start(wx) if wx else 0.0
        py = tr.get("pyannote") or []
        ref = [Segment(p["start"] - skip_s, p["end"] - skip_s, str(p["speaker"]), "")
               for p in py
               if p.get("end", 0) - skip_s > 0 and p.get("start", 1e12) - skip_s < dur]
        print(f"pyannote ref: {len(ref)} turns, {len(set(s.speaker for s in ref))} speakers "
              f"(skip offset {skip_s:.0f}s)", flush=True)

        # --- 1) single pass ---
        t0 = time.time()
        single = moss_transcribe(model, proc, wav, sr, dev)
        single = [s for s in single if not s.text.startswith("[")]
        print(f"single-pass: {len(single)} segs, {len(set(s.speaker for s in single))} speakers "
              f"({time.time()-t0:.0f}s)", flush=True)

        # --- 2) chunked ---
        t0 = time.time()
        chunked_raw: list[Segment] = []
        n_win = 0
        for off in np.arange(0.0, dur, args.window_s):
            piece = wav[int(off * sr):int(min(off + args.window_s, dur) * sr)]
            if len(piece) < sr * 5:
                continue
            segs = moss_transcribe(model, proc, piece, sr, dev)
            for s in segs:
                if not s.text.startswith("["):
                    chunked_raw.append(Segment(s.start + off, s.end + off,
                                               f"w{n_win}:{s.speaker}", s.text))
            n_win += 1
        print(f"chunked: {n_win} windows, {len(chunked_raw)} segs ({time.time()-t0:.0f}s)", flush=True)

        # --- 3) global linking ---
        linked = link_global(wav, sr, chunked_raw, embedder)
        print(f"linked: {len(set(s.speaker for s in linked))} global speakers", flush=True)

        # --- 4) scores ---
        cons_cs = speaker_consistency(single, linked)
        print(f"chunked-vs-singlepass consistency = {cons_cs:.3f}  <-- chunking cost", flush=True)
        if ref:
            for name, hyp in [("single-pass", single), ("chunked+linked", linked)]:
                try:
                    c = speaker_consistency(ref, hyp)
                    d = der(ref, hyp)
                    print(f"vs pyannote: {name:15s} consistency={c:.3f} DER={d:.3f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"vs pyannote: {name} scoring failed: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
