#!/usr/bin/env python
"""Fine-tune MOSS-Transcribe-Diarize on our zh-TW meeting corpus (single GPU).

Targets are built from our exact TTS labels in MOSS's native format:
    [start][Sxx]text[end]...
with Traditional-Chinese text (the corpus is already zh-TW), teaching the model
native Traditional output + TW meeting vocabulary + our domain code-switching.

Standard SFT: audio + prompt -> assistant transcript; CE on assistant tokens only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def target_text(segments) -> str:
    parts = []
    for s in segments:
        spk = int(s["speaker"]) + 1 if str(s["speaker"]).isdigit() else 1
        parts.append(f"[{s['start']:.2f}][S{spk:02d}]{s['text']}[{s['end']:.2f}]")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss")
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-audio-s", type=float, default=120.0, help="clip audio (VRAM)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="models/moss_ft_zhtw")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    rows = [json.loads(l) for l in open(ROOT / args.manifest)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    print(f"{len(rows)} meetings; training {args.steps} steps", flush=True)

    losses = []
    for step in range(args.steps):
        rec = rows[step % len(rows)]
        wav, sr = sf.read(rec["audio_path"])
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1), dtype=np.float32)
        max_n = int(args.max_audio_s * sr)
        clipped = len(wav) > max_n
        if clipped:
            wav = wav[:max_n]
        segs = [s for s in rec["segments"] if s["start"] < args.max_audio_s] if clipped else rec["segments"]
        if not segs:
            continue
        # resample to processor rate if needed
        tgt_sr = proc.feature_extractor.sampling_rate
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        instruction = DEFAULT_PROMPT
        assistant = target_text(segs)
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": rec["audio_path"]},
            {"type": "text", "text": instruction}]}]
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + assistant + proc.tokenizer.eos_token

        enc = proc(text=full_text, audio=[wav], max_length=args.max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        # label mask: only assistant tokens
        prompt_ids = proc(text=prompt_text, audio=[wav], max_length=args.max_len, return_tensors="pt")["input_ids"]
        n_prompt = prompt_ids.shape[1]
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
        batch["labels"] = labels

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        out = model(**batch)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 10 == 0:
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e}", flush=True)

    outdir = ROOT / args.out
    model.save_pretrained(outdir)
    proc.save_pretrained(outdir)
    print(f"\nFT loss {losses[0]:.3f} -> {sum(losses[-10:])/10:.3f} | saved -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
