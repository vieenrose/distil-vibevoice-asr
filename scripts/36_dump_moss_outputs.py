#!/usr/bin/env python
"""Dump MOSS transcriptions + ECAPA embeddings for the real IVOD meetings.

One GPU pass produces everything the linking experiments (34b) and the HF Space
precompute need, so clustering methods/thresholds can be swept offline:

  data/chunk_dump/<stem>.json   segments for single-pass (<=35 min only) and
                                chunked at 300 s and 600 s windows
  data/chunk_dump/<stem>.npz    ECAPA embedding per chunked segment (key
                                "<window_s>/<idx>"; absent = too short)
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

from distil_vibevoice.runtime.embeddings import load_embedder
from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/chunk_dump"
MANIFESTS = ["data/raw/ivod_eval/manifest.jsonl",
             "data/raw/ivod_demo/manifest_long.jsonl"]
WINDOWS = [180.0, 300.0]
SINGLE_PASS_MAX_MIN = 35.0


def moss_transcribe(model, proc, wav, dev, prompt, raw_sink=None):
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=text, audio=[wav], return_tensors="pt").to(dev, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    gen = proc.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    if raw_sink is not None:
        raw_sink.append(gen)
    return parse_transcript_lenient(gen)


def main() -> int:
    from scipy.signal import resample_poly
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v2")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    mdl = str(ROOT / args.model)
    model = AutoModelForCausalLM.from_pretrained(
        mdl, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(mdl, trust_remote_code=True)
    tgt_sr = proc.feature_extractor.sampling_rate
    embedder = load_embedder("ecapa")

    rows = []
    for mpath in MANIFESTS:
        p = ROOT / mpath
        if p.exists():
            rows += [json.loads(l) for l in p.open()]
    seen = set()
    for rec in rows:
        stem = Path(rec["audio_path"]).stem
        if stem in seen:
            continue
        seen.add(stem)
        wav24, sr = sf.read(rec["audio_path"])
        wav24 = np.asarray(wav24 if wav24.ndim == 1 else wav24.mean(1),
                           dtype=np.float32)
        from math import gcd
        g = gcd(sr, tgt_sr)
        wav = resample_poly(wav24, tgt_sr // g, sr // g).astype(np.float32)
        dur = len(wav) / tgt_sr
        print(f"===== {stem}: {dur/60:.1f} min =====", flush=True)

        dump = {"stem": stem, "audio_path": rec["audio_path"],
                "duration": dur, "meta_ref": rec.get("meta", {}),
                "single": None, "chunked": {}}
        embs: dict[str, np.ndarray] = {}

        if dur <= SINGLE_PASS_MAX_MIN * 60:
            t0 = time.time()
            segs = moss_transcribe(model, proc, wav, dev, DEFAULT_PROMPT)
            print(f"  single-pass: {len(segs)} segs ({time.time()-t0:.0f}s)",
                  flush=True)
            dump["single"] = [{"start": s.start, "end": s.end,
                               "speaker": s.speaker, "text": s.text}
                              for s in segs]

        for win_s in WINDOWS:
            t0 = time.time()
            out_segs = []
            raws: list[str] = []
            n_win = 0
            for off in np.arange(0.0, dur, win_s):
                piece = wav[int(off * tgt_sr): int(min(off + win_s, dur) * tgt_sr)]
                if len(piece) < tgt_sr * 2:
                    continue
                for s in moss_transcribe(model, proc, piece, dev, DEFAULT_PROMPT,
                                         raw_sink=raws):
                    end = min(s.end, len(piece) / tgt_sr + 1.0)
                    idx = len(out_segs)
                    out_segs.append({"start": round(off + s.start, 2),
                                     "end": round(off + end, 2),
                                     "win": n_win, "win_speaker": s.speaker,
                                     "text": s.text})
                    e = embedder.embed(
                        piece[int(s.start * tgt_sr): int(end * tgt_sr)], tgt_sr)
                    if float(np.linalg.norm(e)) > 0:
                        embs[f"{int(win_s)}/{idx}"] = e
                n_win += 1
            print(f"  chunked@{int(win_s)}s: {n_win} win, {len(out_segs)} segs "
                  f"({time.time()-t0:.0f}s)", flush=True)
            dump["chunked"][str(int(win_s))] = out_segs
            dump.setdefault("raw", {})[str(int(win_s))] = raws

        (out_dir / f"{stem}.json").write_text(
            json.dumps(dump, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(out_dir / f"{stem}.npz", **embs)
        print(f"  dumped -> {out_dir}/{stem}.json/.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
