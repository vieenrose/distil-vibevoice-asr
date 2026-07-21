#!/usr/bin/env python3
"""Build a MIXED-precision GGUF from the f32 master.

Motivation: uniform q8_0 is 3.7x smaller than f32 and equivalent on English, but
on 16-min Chinese it collapses segmentation -- 69 utterances where f32 emits 312
(58 chars/utt vs 12) -- while still scoring 93% on text. Text metrics do not see
it. If only a few tensor families are responsible, keeping those at f16 recovers
f32 behaviour at most of the size win.

Policy is per tensor-name FAMILY (the per-layer index is stripped), so
`qwen3.blk.*.ffn_down.weight` is one knob rather than 28.

Tensors the engine's own quantiser leaves alone (norms, biases, mel_filters)
stay F32 here too -- matching its allowlist exactly, so a `--policy all=q8_0`
build should reproduce the stock q8_0 file. That equivalence is the self-test.

Usage:
  .venv/bin/python scripts/83_mix_quant.py IN_f32.gguf OUT.gguf \\
      --default q8_0 --f16 token_embd adaptor
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf import quants

# Mirrors is_quantizable_name() in the vendored cli.cpp.
QUANTIZABLE = (
    {"token_embd.weight", "adaptor.fc1.w", "adaptor.fc2.w"}
    | {f"qwen3.blk.{r}" for r in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                                  "attn_o.weight", "ffn_gate.weight", "ffn_up.weight",
                                  "ffn_down.weight")}
    | {f"enc.blk.{r}" for r in ("attn_q.w", "attn_k.w", "attn_v.w", "attn_out.w",
                                "ffn_1.w", "ffn_2.w")}
)

TYPES = {"f32": GGMLQuantizationType.F32, "f16": GGMLQuantizationType.F16,
         "q8_0": GGMLQuantizationType.Q8_0, "q6_k": GGMLQuantizationType.Q6_K,
         "q5_k": GGMLQuantizationType.Q5_K, "q4_k": GGMLQuantizationType.Q4_K}


def family(name: str) -> str:
    """Strip the per-layer index so a family is one knob."""
    return re.sub(r"\.blk\.\d+\.", ".blk.", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--default", default="q8_0", choices=list(TYPES))
    for t in ("f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k"):
        ap.add_argument(f"--{t}", nargs="*", default=[], metavar="FAMILY",
                        help=f"families to force to {t} (substring match)")
    args = ap.parse_args()

    overrides = []          # (substring, type) — first match wins
    for t in ("f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k"):
        for pat in getattr(args, t):
            overrides.append((pat, t))

    r = GGUFReader(args.src)
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
    w = GGUFWriter(args.dst, arch)

    # Carry every KV across untouched. The converter puts the mel filterbank,
    # tokenizer, dims and time-marker params in here; dropping any of it makes a
    # file the loader cannot use.
    from gguf.constants import GGUFValueType
    for key, field in r.fields.items():
        if key in ("general.architecture", "GGUF.version", "GGUF.tensor_count",
                   "GGUF.kv_count"):
            continue
        val = field.contents()
        vt = field.types[0]
        if vt == GGUFValueType.ARRAY:
            sub = field.types[1]
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, vt)

    counts = {}
    for t in r.tensors:
        name = t.name
        fam = family(name)
        if fam not in QUANTIZABLE:
            ttype = "f32"
        else:
            ttype = args.default
            for pat, forced in overrides:
                if pat in fam:
                    ttype = forced
                    break
        data = np.array(t.data)
        # Source is F32 already; de-quantising is never needed here.
        if ttype == "f32":
            out = data.astype(np.float32)
            qt = GGMLQuantizationType.F32
        elif ttype == "f16":
            out = data.astype(np.float16)
            qt = GGMLQuantizationType.F16
        else:
            qt = TYPES[ttype]
            out = quants.quantize(data.astype(np.float32), qt)
        counts[ttype] = counts.get(ttype, 0) + 1
        # Do NOT pass raw_shape: with quantised bytes the writer derives the
        # logical shape itself, and handing it the logical dims makes it try to
        # read them as a byte count ("bytes per row (1024) not a multiple of 34").
        w.add_tensor(name, out, raw_dtype=qt)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.dst}  ({Path(args.dst).stat().st_size/1e9:.2f} GB)")
    print("tensor types:", counts)


if __name__ == "__main__":
    main()
