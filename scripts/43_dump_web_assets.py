#!/usr/bin/env python
"""Dump exact preprocessing assets for the in-browser pipeline.

space_local/models/
  mel.bin       float32: [400 hann window | 201*80 slaney mel filterbank]
  vocab.json    id -> token string (decode-only; GPT-2 byte-level)
  config.json   prompt token ids, special ids, mel geometry

Dumping the filterbank/window from the Python feature extractor (instead of
porting slaney-mel construction to JS) removes the main numerical-mismatch
risk between browser and reference pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_local/models"

PREFIX = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
          151644, 872, 198, 151669]
SUFFIX = [151670, 198, 14880, 44063, 111268, 46670, 61443, 17714, 108704,
          3837, 73157, 104383, 58362, 23031, 71618, 26606, 20450, 111420,
          33108, 104283, 17340, 72640, 9909, 58, 50, 15, 16, 60, 5373, 58,
          50, 15, 17, 60, 5373, 58, 50, 15, 18, 60, 1940, 7552, 111749,
          3837, 110644, 17714, 110019, 105761, 43815, 90395, 18493, 37474,
          100072, 111066, 80565, 20450, 111420, 3837, 23031, 104542, 117932,
          75882, 37474, 105761, 101121, 1773, 151645, 198, 151644, 77091, 198]


def main() -> int:
    import torch
    from transformers import AutoProcessor

    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(
        str(ROOT / "models/moss_ft_zhtw_v2"), trust_remote_code=True)
    fe = proc.feature_extractor

    window = torch.hann_window(fe.n_fft, periodic=True).numpy().astype(np.float32)
    filters = np.asarray(fe.mel_filters, dtype=np.float32)  # [n_freq, n_mel]
    print("window", window.shape, "filters", filters.shape)
    with (OUT / "mel.bin").open("wb") as f:
        f.write(window.tobytes())
        f.write(filters.astype(np.float32).tobytes())

    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    (OUT / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    cfg = {
        "prefix_ids": PREFIX,
        "suffix_ids": SUFFIX,
        "audio_pad_id": 151671,
        "eos_id": 151645,
        "n_fft": int(fe.n_fft),
        "hop": int(fe.hop_length),
        "n_mel": int(fe.feature_size),
        "sr": int(fe.sampling_rate),
        "chunk_frames": 3000,
        "n_freq": int(filters.shape[0]),
        "n_layers": 28,
        "kv_heads": 8,
        "head_dim": 128,
    }
    (OUT / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    print("assets ->", OUT)

    # reference mel for JS validation: 3 s of a real clip
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wav24, sr = sf.read(ROOT / "data/raw/ivod_eval/ivod_2024_15362.wav")
    wav24 = np.asarray(wav24 if wav24.ndim == 1 else wav24.mean(1), np.float32)
    g = gcd(sr, 16000)
    wav = resample_poly(wav24, 16000 // g, sr // g).astype(np.float32)[:16000 * 3]
    m = fe(wav, sampling_rate=16000, return_tensors="np",
           padding="max_length")["input_features"][0]
    ref_dir = ROOT / "data/web_ref"
    ref_dir.mkdir(exist_ok=True)
    wav.tofile(ref_dir / "test3s.f32")
    m.astype(np.float32).tofile(ref_dir / "test3s.mel.f32")
    print("mel ref:", m.shape, "->", ref_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
