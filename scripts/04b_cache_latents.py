#!/usr/bin/env python
"""Cache frozen σ-VAE encoder latents so the corpus scales without raw audio.

The acoustic + semantic tokenizers are FROZEN during distillation, so their
output for a clip never changes — encode once, cache the 7.5 Hz latents, and
train off those (also skips the encoder forward at train time).

Storage: 7.5 tok/s x (64 acoustic + 128 semantic) dims x 2 bytes (fp16)
       ~= 2.88 KB/s ~= 10.4 MB/hour  ->  ~100 GB for 10,000 hours
vs raw 24 kHz mono WAV at 172.8 MB/hour (~1.7 TB) — ~17x smaller.

With --delete-audio, removes each wav after caching (stream-and-discard), so
peak disk stays at one batch of audio regardless of total corpus size.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
CHUNK_S = 60.0  # encode in <=60 s chunks (conv overflow guard)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/teacher")
    ap.add_argument("--audio-glob", default="data/raw/ivod/*.wav")
    ap.add_argument("--out", default="data/latents")
    ap.add_argument("--delete-audio", action="store_true", help="remove wav after caching (stream-and-discard)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

    outdir = Path(ROOT / args.out); outdir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.teacher, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    acoustic = model.model.acoustic_tokenizer
    semantic = model.model.semantic_tokenizer

    paths = sorted(glob.glob(str(ROOT / args.audio_glob)))
    print(f"{len(paths)} audio files -> {outdir}")
    tot_audio_mb = tot_lat_mb = tot_hours = 0.0
    done = 0
    for p in paths:
        stem = Path(p).stem
        cache = outdir / f"{stem}.npz"
        if cache.exists():
            continue
        wav, sr = sf.read(p)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:  # collector already emits 24k, but be safe
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, SR); wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        dur_h = len(wav) / SR / 3600
        ac_means, se_means = [], []
        with torch.no_grad():
            step = int(CHUNK_S * SR)
            for s0 in range(0, len(wav), step):
                chunk = torch.tensor(wav[s0:s0 + step], device=args.device, dtype=dtype).unsqueeze(0).unsqueeze(1)
                if chunk.shape[-1] < SR // 10:
                    continue
                ac_means.append(acoustic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
                se_means.append(semantic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
        if not ac_means:
            continue
        ac = np.concatenate(ac_means, axis=0)  # (T, 64)
        se = np.concatenate(se_means, axis=0)  # (T, 128)
        np.savez_compressed(cache, acoustic=ac, semantic=se, sr=SR, hop=3200)
        audio_mb = os.path.getsize(p) / 1e6
        lat_mb = cache.stat().st_size / 1e6
        tot_audio_mb += audio_mb; tot_lat_mb += lat_mb; tot_hours += dur_h; done += 1
        print(f"  {stem}: {dur_h*60:.1f} min | audio {audio_mb:.1f}MB -> latents {lat_mb:.1f}MB "
              f"({audio_mb/max(lat_mb,1e-6):.1f}x), shapes ac{ac.shape} se{se.shape}")
        if args.delete_audio:
            os.remove(p)

    if done:
        print(f"\ncached {done} files, {tot_hours:.2f} h | audio {tot_audio_mb/1000:.2f}GB -> "
              f"latents {tot_lat_mb/1000:.3f}GB ({tot_audio_mb/max(tot_lat_mb,1e-6):.1f}x smaller)")
        print(f"projected 10,000 h latents: {tot_lat_mb/max(tot_hours,1e-6)*10000/1000:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
