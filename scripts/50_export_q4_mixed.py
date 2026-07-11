"""Mixed-precision q4 export: q4 (MatMulNBits) for most decoder MatMuls, int8
for the q4-sensitive groups (from scripts/48). Keeps download small while
protecting the layers that hurt most under q4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="models/moss_onnx_v4qat2/decoder.onnx")
    ap.add_argument("--out", default="models/moss_web_v4/decoder.q4mixed.onnx")
    ap.add_argument("--int8-frags", nargs="+", default=["layers.27", "lm_head"],
                    help="substrings of MatMul node names to keep at int8")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        MatMul4BitsQuantizer,
    )

    model = onnx.load(args.decoder)
    # node names of the MatMuls we want at int8 (excluded from q4)
    int8_nodes = [n.name for n in model.graph.node
                  if n.op_type == "MatMul"
                  and any(f in n.name for f in args.int8_frags)]
    print(f"int8 MatMuls ({len(int8_nodes)}): "
          f"{[n[-40:] for n in int8_nodes[:4]]}...")

    # 1) q4 everything EXCEPT the int8 nodes (they stay fp32 for now)
    q = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True,
                             nodes_to_exclude=int8_nodes)
    q.process()
    tmp = args.out + ".q4only.onnx"
    onnx.save(q.model.model, tmp)

    # 2) int8-quantize ONLY the excluded MatMuls (+ any Gather) in that graph
    quantize_dynamic(tmp, args.out, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"],
                     nodes_to_quantize=int8_nodes)
    Path(tmp).unlink(missing_ok=True)
    print(f"{args.out}: {Path(args.out).stat().st_size // 2**20} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
