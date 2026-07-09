#!/usr/bin/env python
"""Augment TTS meetings + cache σ-VAE latents (GPU1). Labels untouched.

Per meeting: read wav, with prob --aug-frac apply augment_wav (RIR reverb + MUSAN
noise + codec sim), then encode frozen acoustic+semantic latents (fp16 npz).
Augmentation only changes acoustics, so the exact TTS labels stay valid.
Writes a manifest row with latents_path in meta. Sharded for parallelism.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
CHUNK_S = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/teacher")
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--latents-out", default="data/latents/tts")
    ap.add_argument("--out-manifest", default="data/pseudo/tts_cached.jsonl")
    ap.add_argument("--rir-dir", default="data/aug/RIRS_NOISES")
    ap.add_argument("--musan-dir", default="data/aug/musan")
    ap.add_argument("--aug-frac", type=float, default=0.75)
    ap.add_argument("--device", default="cuda:0")  # note: set CUDA_VISIBLE_DEVICES=1
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    from distil_vibevoice.data.augment import augment_wav
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

    latdir = Path(ROOT / args.latents_out); latdir.mkdir(parents=True, exist_ok=True)
    mpath = Path(ROOT / f"{args.out_manifest}.shard{args.shard}")
    done = set()
    if mpath.exists():
        for l in mpath.read_text().splitlines():
            try: done.add(json.loads(l)["meta"]["src"])
            except Exception: pass

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    acoustic = model.model.acoustic_tokenizer
    semantic = model.model.semantic_tokenizer
    dtype = next(model.parameters()).dtype

    rows = [json.loads(l) for l in open(ROOT / args.manifest)]
    mf = mpath.open("a", encoding="utf-8")
    n = tot_h = 0.0
    for i, rec in enumerate(rows):
        if args.num_shards > 1 and i % args.num_shards != args.shard:
            continue
        src = rec["audio_path"]
        if src in done or not os.path.exists(src):
            continue
        wav, sr = sf.read(src)
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1), dtype=np.float32)
        # deterministic augment decision per file
        rng = np.random.default_rng(zlib.crc32(src.encode()))
        augmented = rng.random() < args.aug_frac
        if augmented:
            wav = augment_wav(wav, SR, rir_dir=args.rir_dir, musan_dir=args.musan_dir,
                              codec_prob=0.4, snr_db_range=(5.0, 25.0), rng=rng)
        ac_m, se_m = [], []
        with torch.no_grad():
            step = int(CHUNK_S * SR)
            for s0 in range(0, len(wav), step):
                chunk = torch.tensor(wav[s0:s0 + step], device=args.device, dtype=dtype).unsqueeze(0).unsqueeze(1)
                if chunk.shape[-1] < SR // 10:
                    continue
                ac_m.append(acoustic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
                se_m.append(semantic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
        if not ac_m:
            continue
        stem = Path(src).stem + (f"_aug{args.shard}" if augmented else "")
        latfile = latdir / f"{stem}.npz"
        np.savez_compressed(latfile, acoustic=np.concatenate(ac_m), semantic=np.concatenate(se_m), sr=SR, hop=3200)
        rec2 = dict(rec)
        rec2["meta"] = {**rec.get("meta", {}), "latents_path": str(latfile.relative_to(ROOT)),
                        "augmented": bool(augmented), "src": src}
        mf.write(json.dumps(rec2, ensure_ascii=False) + "\n"); mf.flush()
        n += 1; tot_h += rec["duration_s"] / 3600
        if int(n) % 50 == 0:
            print(f"  shard {args.shard}: {int(n)} cached, {tot_h:.1f}h", flush=True)
    print(f"\nshard {args.shard}: {int(n)} meetings cached, {tot_h:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
