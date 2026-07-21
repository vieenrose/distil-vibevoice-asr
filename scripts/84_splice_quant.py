#!/usr/bin/env python3
"""Splice a mixed-precision GGUF from pre-built FULL-precision source files.

83_mix_quant.py quantizes in Python via `gguf.quants.quantize`, which has no
implementation for the K-quants (Q4_K/Q5_K/Q6_K raise NotImplementedError --
only the legacy Q4_0/Q5_0/Q8_0 paths exist in this version of the package).

This script sidesteps that: it takes ALREADY-QUANTIZED full GGUF files (built
with the vendored `rs-moss-td quantize` CLI, which supports every K-quant) and
copies each tensor's raw bytes verbatim from whichever source file the policy
assigns it to. No floating-point requantization happens in Python at all, so
whatever accuracy the C++ quantizer achieves is exactly what ends up in the
spliced file.

Usage:
  ./build-cuda/rs-moss-td quantize IN_f32.gguf full_q4k.gguf q4_k
  .venv/bin/python scripts/84_splice_quant.py OUT.gguf \\
      --default full_q4k.gguf=q4_k \\
      --override token_embd,qwen3.blk=IN_f32.gguf=f16
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType


def family(name: str) -> str:
    return re.sub(r"\.blk\.\d+\.", ".blk.", name)


def load(path: str):
    r = GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--default", required=True,
                    help="PATH=LABEL, e.g. full_q4k.gguf=q4_k")
    ap.add_argument("--override", nargs="*", default=[],
                    help="FAMILY1,FAMILY2=PATH=LABEL; first match wins over --default")
    args = ap.parse_args()

    def parse_spec(s: str):
        path, label = s.rsplit("=", 1)
        return path, label

    default_path, default_label = parse_spec(args.default)
    overrides = []   # (families, path, label)
    for spec in args.override:
        fams, path, label = spec.split("=")
        overrides.append((fams.split(","), path, label))

    sources = {}   # path -> (reader, {name: tensor})
    def get_source(path):
        if path not in sources:
            sources[path] = load(path)
        return sources[path]

    base_reader, base_tensors = get_source(default_path)
    arch = str(bytes(base_reader.fields["general.architecture"].parts[-1]), "utf-8")
    w = GGUFWriter(args.out, arch)

    # KV comes from the f32 master's own converter output, which is identical
    # across every quantized derivative (the CLI quantizer only touches tensor
    # data) -- so the default source's KV is authoritative.
    for key, field in base_reader.fields.items():
        if key in ("general.architecture", "GGUF.version", "GGUF.tensor_count",
                   "GGUF.kv_count"):
            continue
        val, vt = field.contents(), field.types[0]
        if vt == GGUFValueType.ARRAY:
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, vt)

    counts = {}
    for name, base_t in base_tensors.items():
        fam = family(name)
        path, label = default_path, default_label
        for fams, p, lb in overrides:
            if any(f in fam for f in fams):
                path, label = p, lb
                break
        _, tensors = get_source(path)
        t = tensors[name]
        counts[label] = counts.get(label, 0) + 1
        w.add_tensor(name, t.data, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e9:.2f} GB)")
    print("tensor labels:", counts)


if __name__ == "__main__":
    main()
