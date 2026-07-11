#!/usr/bin/env python
"""FT v3-long: make long-window behavior robust on real far-field audio.

Fixes the validated long-clip regressions of v2 (whole windows flipping to
Simplified, [Sxx]-tag drops) at the model level:

  1. TRAIN AT INFERENCE LENGTH: 300 s windows (v2 capped audio at 120 s, so
     300 s inference windows were out of the trained length distribution).
  2. ACOUSTIC ROBUSTNESS: on-the-fly far-field augmentation (RIR + MUSAN
     noise + codec sim) of the exact-label TTS corpus.
  3. REAL FAR-FIELD DATA: IVOD meetings with whisperx x pyannote fused
     targets (scripts/38), sampled as random 300 s windows.

Per-step sampling: p_ivod real | (1-p_ivod) TTS, of which p_aug augmented.
Windows renumber speakers by first appearance (MOSS convention: S01 = first
voice heard in the window). Recipe otherwise = v2 (SFT, CE on assistant
tokens, bf16, grad ckpt, cosine LR).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def window_sample(rec: dict, max_s: float, rng: random.Random):
    """Pick a random max_s window; return (offset, segments shifted+renumbered)."""
    dur = float(rec.get("duration") or max(s["end"] for s in rec["segments"]))
    for _ in range(8):
        off = 0.0 if dur <= max_s else rng.uniform(0.0, dur - max_s)
        inside = [s for s in rec["segments"]
                  if s["start"] >= off and s["end"] <= off + max_s]
        if len(inside) >= 2 or dur <= max_s:
            break
    if not inside:
        return None, None
    spk_map: dict = {}
    segs = []
    for s in sorted(inside, key=lambda x: x["start"]):
        key = s["speaker"]
        if key not in spk_map:
            spk_map[key] = len(spk_map) + 1
        segs.append({"start": s["start"] - off, "end": s["end"] - off,
                     "speaker": spk_map[key], "text": s["text"]})
    return off, segs


def target_text(segments) -> str:
    return "".join(f"[{s['start']:.2f}][S{s['speaker']:02d}]{s['text']}"
                   f"[{s['end']:.2f}]" for s in segments)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v2")
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.3)
    ap.add_argument("--p-aug", type=float, default=0.43,
                    help="P(augment | TTS sample) -> ~30%% of all steps")
    ap.add_argument("--rir-dir", default="data/aug/RIRS_NOISES")
    ap.add_argument("--musan-dir", default="data/aug/musan")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-audio-s", type=float, default=300.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default="models/moss_ft_zhtw_v3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    from distil_vibevoice.data.augment import augment_wav

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    tts_rows = []
    for mp in args.tts_manifests:
        p = ROOT / mp
        if p.exists():
            tts_rows += [json.loads(l) for l in p.open()]
    ivod_rows = []
    p = ROOT / args.ivod_manifest
    if p.exists():
        ivod_rows = [json.loads(l) for l in p.open()]
    if not ivod_rows:
        print("WARNING: no IVOD rows; falling back to TTS-only (+aug)")
        args.p_ivod = 0.0
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rng.shuffle(tts_rows)
    rng.shuffle(ivod_rows)
    print(f"TTS {len(tts_rows)} | IVOD {len(ivod_rows)} meetings; "
          f"{args.steps} steps @ {args.max_audio_s:.0f}s windows", flush=True)

    tgt_sr = proc.feature_extractor.sampling_rate
    losses, n_skip = [], 0
    ti = ii = 0
    for step in range(args.steps):
        use_ivod = ivod_rows and rng.random() < args.p_ivod
        if use_ivod:
            rec = ivod_rows[ii % len(ivod_rows)]
            ii += 1
        else:
            rec = tts_rows[ti % len(tts_rows)]
            ti += 1

        off, segs = window_sample(rec, args.max_audio_s, rng)
        if not segs:
            n_skip += 1
            continue
        try:
            wav, sr = sf.read(rec["audio_path"],
                              start=int(off * 24000),
                              frames=int(args.max_audio_s * 24000))
        except Exception:
            wav, sr = sf.read(rec["audio_path"])
            wav = wav[int(off * sr): int((off + args.max_audio_s) * sr)]
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1),
                         dtype=np.float32)
        if len(wav) < sr:
            n_skip += 1
            continue

        aug = (not use_ivod) and rng.random() < args.p_aug
        if aug:
            wav = augment_wav(wav, sr, rir_dir=args.rir_dir,
                              musan_dir=args.musan_dir, rng=np_rng)

        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        assistant = target_text(segs)
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        prompt_text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + assistant + proc.tokenizer.eos_token

        enc = proc(text=full_text, audio=[wav], max_length=args.max_len,
                   return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        n_prompt = proc(text=prompt_text, audio=[wav],
                        max_length=args.max_len,
                        return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in enc.items()}
        batch["labels"] = labels

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        out = model(**batch)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 10 == 0:
            src = "ivod" if use_ivod else ("tts+aug" if aug else "tts")
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e} "
                  f"[{src}] len={input_ids.shape[1]}", flush=True)
        if args.save_every and step and step % args.save_every == 0:
            ck = ROOT / (args.out + f"_ckpt{step}")
            model.save_pretrained(ck)
            proc.save_pretrained(ck)
            print(f"  ckpt -> {ck}", flush=True)

    outdir = ROOT / args.out
    model.save_pretrained(outdir)
    proc.save_pretrained(outdir)
    print(f"\nloss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f} "
          f"(skipped {n_skip}) | saved -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
