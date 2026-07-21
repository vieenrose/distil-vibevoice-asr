#!/usr/bin/env python
"""Margin probe at GROUND-TRUTH boundaries (non-gameable).

Why this exists: the first margin metric sampled positions where the MODEL
chose to emit '['. v7.3 scored a median +7.15 (better than base's +4.90) while
emitting 114 markers on audio with 12 ground-truth utterances and collapsing to
49 characters through ggml. A metric that improves as the model degrades is
worse than no metric -- it nearly certified a broken model.

Here the positions come from the reference segmentation, teacher-forced, so the
model cannot inflate the score by over-emitting:

  POSITIVE margin: at a true boundary, logit('[') - best competing token.
                   Low  -> the marker decision is a coin flip under ggml/quant
                           noise (this is the marker-collapse failure).
  NEGATIVE margin: at ordinary text positions, logit(gold) - logit('[').
                   Low  -> the model is tempted to open a marker mid-sentence
                           (this is the v7.3 marker-spam failure).

A healthy model needs BOTH comfortably positive. Reporting only the first is
what let v7.3 look like a success.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_s55():
    spec = importlib.util.spec_from_file_location(
        "s55", str(ROOT / "scripts/55_v61_marker_ft.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--audio-stem", default="ivod_2024_15804.wav")
    ap.add_argument("--offset", type=float, default=600.0)
    ap.add_argument("--dur", type=float, default=180.0)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    s55 = load_s55()

    rec = None
    for line in open(ROOT / args.manifest):
        if args.audio_stem in line:
            rec = json.loads(line)
            break
    if rec is None:
        print("audio not found in manifest")
        return 1
    segs = [dict(s, start=s["start"] - args.offset, end=s["end"] - args.offset)
            for s in rec["segments"]
            if args.offset <= s["start"] and s["end"] <= args.offset + args.dur]
    segs = [{**s, "speaker": 1} for s in segs]
    dense = s55.densify_segments(segs)
    target = s55.target_text(dense)
    print(f"reference: {len(segs)} raw -> {len(dense)} densified boundaries "
          f"over {args.dur:.0f}s")

    wav, sr = sf.read(rec["audio_path"], start=int(args.offset * 24000),
                      frames=int(args.dur * 24000))
    wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)

    dev = torch.device(args.device)
    print(f"\n{'model':34s} {'POS margin (at true bounds)':>28s} "
          f"{'NEG margin (text vs [)':>24s}")
    for mp in args.models:
        path = str(ROOT / mp) if not mp.startswith("/") else mp
        p = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        tok = p.tokenizer
        m = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        prompt = p.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        enc = p(text=prompt + target + tok.eos_token, audio=[wav],
                return_tensors="pt").to(dev, torch.float32)
        n_prompt = p(text=prompt, audio=[wav],
                     return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = m(**enc).logits[0].float()
        ids = enc["input_ids"][0]
        br = tok.encode("[", add_special_tokens=False)[0]

        pos, neg = [], []
        for i in range(n_prompt, ids.shape[0]):
            row = logits[i - 1]
            gold = ids[i].item()
            if gold == br:                       # a true boundary
                r = row.clone(); r[br] = float("-inf")
                pos.append(row[br].item() - r.max().item())
            else:                                # ordinary text
                neg.append(row[gold].item() - row[br].item())
        name = Path(mp).name
        print(f"{name:34s} "
              f"n={len(pos):3d} med {st.median(pos):+6.2f} "
              f"<0.5 {sum(1 for x in pos if x < 0.5):3d} | "
              f"n={len(neg):4d} med {st.median(neg):+6.2f} "
              f"<0.5 {sum(1 for x in neg if x < 0.5):4d}")
        del m
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
