#!/usr/bin/env python
"""Bulk TTS meeting generation with EXACT labels (turn-by-turn), sharded per GPU.

Per script: cap to 4 speakers, clone each to a CV zh-TW voice, generate each turn
as a single-speaker utterance, measure its duration -> exact Segment(start,end,
speaker,text). Concatenate turns with small gaps -> meeting wav + MeetingRecord.
Labels are exact by construction (from the script), no transcription.

Run TWO shards (GPU0/GPU1) for ~2x throughput:
  CUDA_VISIBLE_DEVICES=0 python scripts/24_bulk_tts.py --shard 0 --num-shards 2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/24_bulk_tts.py --shard 1 --num-shards 2 &
Use the TTS venv: .venv_tts + PYTHONPATH=vendor/VibeVoice-community
"""
from __future__ import annotations

import argparse
import glob
import json
import zlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
GAP_S = 0.35  # silence between turns


def load_voice(p: str) -> np.ndarray:
    w, s = sf.read(p)
    w = np.asarray(w if w.ndim == 1 else w.mean(1), dtype=np.float32)
    if s != SR:
        from scipy.signal import resample_poly
        g = gcd(s, SR); w = resample_poly(w, SR // g, s // g).astype(np.float32)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tts_1p5b")
    ap.add_argument("--scripts", default="data/scripts/scripts.jsonl")
    ap.add_argument("--out-wav", default="data/synthetic")
    ap.add_argument("--out-manifest", default="data/pseudo/tts_meetings.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=4)
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    args = ap.parse_args()

    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    outdir = Path(ROOT / args.out_wav); outdir.mkdir(parents=True, exist_ok=True)
    mpath = Path(ROOT / f"{args.out_manifest}.shard{args.shard}")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if mpath.exists():
        for l in mpath.read_text().splitlines():
            try: done.add(json.loads(l)["meta"]["script_idx"])
            except Exception: pass

    proc = VibeVoiceProcessor.from_pretrained(args.model)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    model.set_ddpm_inference_steps(num_steps=10)

    voices_bank = sorted(glob.glob(str(ROOT / "data/raw/common_voice_zhtw/**/*.mp3"), recursive=True))
    scripts = [json.loads(l) for l in open(ROOT / args.scripts)]

    mf = mpath.open("a", encoding="utf-8")
    n_done = tot_h = 0.0
    for idx, sc in enumerate(scripts):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        if idx in done:
            continue
        speakers = sc["speakers"][: args.max_speakers]
        spk_set = set(speakers)
        turns = [t for t in sc["turns"] if t["speaker"] in spk_set]
        if len(turns) < 2:
            continue
        # deterministic voice per speaker
        rng = np.random.default_rng(zlib.crc32(("|".join(speakers)).encode()))
        vmap = {s: voices_bank[int(rng.integers(len(voices_bank)))] for s in speakers}
        vcache = {s: load_voice(vmap[s]) for s in speakers}
        spk_num = {s: i + 1 for i, s in enumerate(speakers)}

        audio_parts, segments, t_cursor = [], [], 0.0
        gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
        for turn in turns:
            spk, text = turn["speaker"], turn["text"].strip()
            if not text:
                continue
            script_line = f"Speaker {spk_num[spk]}: {text}"
            inp = proc(text=[script_line], voice_samples=[[vcache[spk]]], padding=True,
                       return_tensors="pt", return_attention_mask=True)
            inp = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in inp.items()}
            try:
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=None, cfg_scale=args.cfg_scale,
                                         tokenizer=proc.tokenizer, generation_config={"do_sample": False},
                                         verbose=False, is_prefill=True)
                a = out.speech_outputs[0].float().cpu().numpy().reshape(-1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            dur = len(a) / SR
            segments.append({"start": round(t_cursor, 3), "end": round(t_cursor + dur, 3),
                             "speaker": str(spk_num[spk] - 1), "text": text})
            audio_parts += [a, gap]
            t_cursor += dur + GAP_S

        if not segments:
            continue
        wav = np.concatenate(audio_parts).astype(np.float32)
        stem = f"tts_{args.shard}_{idx:04d}"
        wav_path = outdir / f"{stem}.wav"
        sf.write(wav_path, wav, SR)
        rec = {"audio_path": str(wav_path), "duration_s": round(len(wav) / SR, 2),
               "sample_rate": SR, "language": "zh-TW-en", "source": "tts_synthetic",
               "split": "train", "segments": segments,
               "meta": {"script_idx": idx, "n_speakers": len(speakers), "domain": sc.get("domain", "")}}
        mf.write(json.dumps(rec, ensure_ascii=False) + "\n"); mf.flush()
        n_done += 1; tot_h += len(wav) / SR / 3600
        print(f"  [{stem}] {len(segments)} turns, {len(wav)/SR/60:.1f}min, {len(speakers)}spk", flush=True)

    print(f"\nshard {args.shard}: {int(n_done)} meetings, {tot_h:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
