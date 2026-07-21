#!/usr/bin/env python3
"""Run the GENUINE MOSS-Transcribe-Diarize PyTorch pipeline on a wav.

This is the reference side of the stage-1 purity gate: whatever this prints is
what a clean C++ port must reproduce byte for byte (transcript text, timestamps
and [Sxx] speaker tags alike).

Deliberately NON-WINDOWED and unmodified: no chunking, no repetition penalty,
no EOS suppression, no post-processing, no s2tw. The audio goes in whole and the
model's raw `[start][Sxx]text[end]` stream comes out. Anything clever added here
would silently become part of the "reference" and defeat the purpose.

Usage:
  .venv/bin/python scripts/80_official_reference.py AUDIO.wav -o out.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
# The `moss_transcribe_diarize` package ships in the authors' GitHub repo, not
# in the HF model repo (the HF repo only carries the remote-code modules).
DEFAULT_PKG = Path("/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="Generation cap. Must be high enough that the model "
                         "stops on EOS by itself -- a cap that truncates would "
                         "make the reference a function of this flag.")
    ap.add_argument("--pkg", type=Path, default=DEFAULT_PKG,
                    help="checkout of github.com/OpenMOSS/MOSS-Transcribe-Diarize")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sdpa-backend", default=None,
                    choices=("efficient", "flash", "math"),
                    help="Force a torch SDPA backend. Default lets torch pick, "
                         "which on long audio picks MATH and materialises the "
                         "full seq x seq attention matrix (11.4 GiB for a 16-min "
                         "clip -> OOM on a 32 GB card). 'efficient' avoids that. "
                         "Only legitimate if it reproduces the 5-min f32 "
                         "reference byte-for-byte -- verify before trusting it.")
    ap.add_argument("--max-memory-per-gpu", default=None,
                    help="e.g. '12GiB'; forces accelerate to actually split "
                         "layers across GPUs instead of packing one card.")
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'balanced' to shard across BOTH GPUs. Needed for "
                         "f32 on long audio: a 16-min clip OOMs on one 32 GB "
                         "card at a single 11.4 GB SDPA attention allocation, "
                         "but fits across 64 GB. f32 is the only reference "
                         "dtype, so this is how long-form parity gets gated.")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "f16", "f32"),
                    help="Reference dtype. bf16 is what the authors' README "
                         "runs on CUDA, but bf16 has an 8-bit mantissa vs "
                         "f16's 10, so a bf16 reference and an f16 port can "
                         "legitimately diverge at near-ties. f32 is the "
                         "dtype-neutral reference to gate a port against.")
    args = ap.parse_args()

    sys.path.insert(0, str(args.pkg))
    from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402
    from moss_transcribe_diarize.inference_utils import (  # noqa: E402
        build_transcription_messages,
        generate_transcription,
    )

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16,
             "f32": torch.float32}[args.dtype]
    if device.type != "cuda":
        dtype = torch.float32

    t0 = time.time()
    if args.device_map:
        # Sharded: accelerate places layers across the visible GPUs and moves
        # activations between them, so no single card has to hold the whole
        # attention matrix.
        # max_memory is what actually forces a SPLIT. Without it "balanced"
        # notes the 3.6 GB f32 model fits on one card and puts it all there --
        # which leaves the big per-layer SDPA attention allocation on that same
        # card and OOMs exactly as before. Capping each GPU below the model size
        # makes accelerate distribute the layers for real.
        mm = None
        if args.max_memory_per_gpu:
            n = torch.cuda.device_count()
            mm = {i: args.max_memory_per_gpu for i in range(n)}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=dtype,
            device_map=args.device_map, max_memory=mm,
        ).eval()
        device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype="auto",
        ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[ref] loaded in {time.time() - t0:.1f}s dtype={dtype} dev={device}",
          file=sys.stderr)

    messages = build_transcription_messages(args.audio)
    t1 = time.time()
    if args.sdpa_backend:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = {"efficient": SDPBackend.EFFICIENT_ATTENTION,
                   "flash": SDPBackend.FLASH_ATTENTION,
                   "math": SDPBackend.MATH}[args.sdpa_backend]
        with sdpa_kernel(backend):
            result = generate_transcription(
                model, processor, messages, max_new_tokens=args.max_new_tokens)
    else:
        result = generate_transcription(
            model, processor, messages, max_new_tokens=args.max_new_tokens)
    wall = time.time() - t1

    # generate_transcription returns {'text': ..., 'generated_tokens': N}.
    # str()-ing that dict silently makes the "reference" a Python repr, which
    # can never match a C++ transcript -- it looks like a parity failure but is
    # purely a bug on this side.
    if isinstance(result, dict):
        text = result.get("text", "")
        print(f"[ref] generated_tokens={result.get('generated_tokens')}",
              file=sys.stderr)
    else:
        text = str(result)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"[ref] {wall:.1f}s -> {args.output} ({len(text)} chars)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
