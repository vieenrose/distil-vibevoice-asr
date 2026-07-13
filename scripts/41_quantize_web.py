#!/usr/bin/env python
"""Quantize the v2 ONNX graphs for the in-browser demo.

  encoder.onnx   (1.2 GB fp32) -> int8 dynamic          (304 MB)
  embedding.onnx (594 MB fp32) -> int8 dynamic (Gather) (149 MB)
  decoder.onnx   (fp32 + ext)  -> q4 MatMulNBits        (357 MB)

Encoder/embedding are consolidated to single files first (onnxruntime-web
fetches one URL per graph). The decoder is quantized straight from its
external-data form — consolidating its ~1.8 GB fp32 first would exceed the
2 GB protobuf serialization limit; the q4 result fits easily.

Idempotent: existing outputs are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]


def consolidate(src: Path, dst: Path) -> None:
    m = onnx.load(str(src))
    onnx.save(m, str(dst))
    print(f"  consolidated {src.name}: {dst.stat().st_size/1e6:.0f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/moss_onnx_v2")
    ap.add_argument("--out", default="models/moss_web")
    ap.add_argument("--q4-block", type=int, default=32)
    args = ap.parse_args()
    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        MatMul4BitsQuantizer,
    )

    if not (out / "encoder.int8.onnx").exists():
        enc_fp32 = tmp / "encoder.fp32.onnx"
        consolidate(src / "encoder.onnx", enc_fp32)
        # MatMul-only: quantizing Conv produces ConvInteger, which onnxruntime
        # CPU/web has no kernel for -> "Could not find an implementation for
        # ConvInteger" at session-create in the browser. The Whisper encoder's
        # convs stay fp32; only its MatMuls are int8'd (still ~300MB, web-safe).
        quantize_dynamic(str(enc_fp32), str(out / "encoder.int8.onnx"),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
    print(f"encoder.int8.onnx: {(out/'encoder.int8.onnx').stat().st_size/1e6:.0f} MB")

    if not (out / "embedding.int8.onnx").exists():
        emb_fp32 = tmp / "embedding.fp32.onnx"
        consolidate(src / "embedding.onnx", emb_fp32)
        quantize_dynamic(str(emb_fp32), str(out / "embedding.int8.onnx"),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=["Gather", "MatMul"])
    print(f"embedding.int8.onnx: {(out/'embedding.int8.onnx').stat().st_size/1e6:.0f} MB")

    if not (out / "decoder.q4.onnx").exists():
        model = onnx.load(str(src / "decoder.onnx"))
        q = MatMul4BitsQuantizer(model, block_size=args.q4_block,
                                 is_symmetric=True)
        q.process()
        onnx.save(q.model.model, str(out / "decoder.q4.onnx"))
    print(f"decoder.q4.onnx: {(out/'decoder.q4.onnx').stat().st_size/1e6:.0f} MB")

    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
