#!/usr/bin/env python
"""QAT-finalize the 1.5B student to int4 (torchao), export for mobile, gate on RAM.

Loads the stage-2 checkpoint from configs/qat_export.yaml, converts Linear
layers to int4-weight/int8-activation with torchao (finalizing QAT
fake-quant modules when the checkpoint was QAT-prepared, otherwise falling
back to post-training quantization — logged), saves the quantized state dict
under export/, and prints the runtime.ram_budget breakdown. Exits 1 if the
estimated total exceeds ``export.max_total_gb`` (the export gate).

TODO-verify (needs real weights + target runtime decision): the final mobile
container — GGUF via llama.cpp convert scripts vs ExecuTorch .pte — is marked
TODO-verify in configs/qat_export.yaml; this script always writes the torchao
.pt artifact and prints conversion instructions for the chosen format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.runtime.ram_budget import estimate_ram


def quantize_int4(model: "object", quant_cfg: dict) -> str:
    """int4-weight/int8-act quantization in place; returns the mode used."""
    import torch
    try:
        from torchao.quantization import Int8DynamicActivationInt4WeightConfig, quantize_
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchao is required for script 10 (pip install torchao)") from exc

    skip = tuple(quant_cfg.get("skip_modules", ["lm_head", "embed_tokens"]))

    def keep(mod: "torch.nn.Module", name: str) -> bool:
        return isinstance(mod, torch.nn.Linear) and not any(s in name for s in skip)

    base = Int8DynamicActivationInt4WeightConfig(group_size=int(quant_cfg.get("group_size", 128)))
    try:  # finalize QAT fake-quant modules if the checkpoint was QAT-prepared
        from torchao.quantization.qat import QATConfig
        quantize_(model, QATConfig(base, step="convert"), filter_fn=keep)
        return "qat-finalize"
    except Exception as exc:
        print(f"[info] QAT convert not applicable ({type(exc).__name__}); "
              "falling back to post-training quantize_", file=sys.stderr)
        quantize_(model, base, filter_fn=keep)
        return "ptq"


def export_artifact(model: "object", export_dir: Path, fmt: str) -> Path:
    import torch
    export_dir.mkdir(parents=True, exist_ok=True)
    pt = export_dir / "student_int4.pt"
    torch.save(model.state_dict(), pt)
    model.config.save_pretrained(export_dir)
    size_gb = pt.stat().st_size / 1e9
    print(f"quantized state dict: {pt} ({size_gb:.2f} GB)")
    if fmt != "pt":
        print(f"[TODO-verify] '{fmt}' container not produced here — runtime choice "
              "(llama.cpp GGUF vs ExecuTorch .pte) is marked TODO-verify in "
              "configs/qat_export.yaml. GGUF: llama.cpp convert_hf_to_gguf.py on the "
              "bf16 checkpoint + its int4 requant; ExecuTorch: export via optimum-executorch.")
    return pt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/qat_export.yaml"))
    ap.add_argument("--source", default=None, help="override model.source checkpoint")
    ap.add_argument("--skip-quant", action="store_true",
                    help="only run the RAM-budget gate (no model load)")
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    mdl, quant, exp = cfg["model"], cfg.get("quant") or {}, cfg.get("export") or {}
    src = ROOT / (args.source or mdl["source"])
    export_dir = ROOT / mdl.get("export_dir", "export/1p5b_int4")

    if not args.skip_quant:
        import torch
        from transformers import Qwen2ForCausalLM
        print(f"loading student from {src}")
        model = Qwen2ForCausalLM.from_pretrained(src, dtype=torch.bfloat16).eval()
        mode = quantize_int4(model, quant)
        print(f"quantization mode: {mode}")
        artifact = export_artifact(model, export_dir, exp.get("format", "gguf"))
        (export_dir / "export_meta.json").write_text(json.dumps(
            {"source": str(src), "mode": mode, "quant": quant,
             "artifact": artifact.name}, indent=2))

    budget = estimate_ram(**(exp.get("ram_budget") or {}))
    print("\nRAM budget (runtime.ram_budget.estimate_ram):")
    for key, gb in budget.items():
        if key != "total_gb":
            print(f"  {key:<24}{gb:>8.3f} GB")
    total = float(budget["total_gb"])
    ceiling = float(exp.get("max_total_gb", 2.4))
    print(f"  {'total_gb':<24}{total:>8.3f} GB  (ceiling {ceiling:.2f} GB)")
    if total > ceiling:
        print(f"EXPORT GATE FAILED: {total:.3f} GB > {ceiling:.2f} GB", file=sys.stderr)
        return 1
    print("export gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
