#!/usr/bin/env python
"""Extract the Qwen2 LLM backbone from a VibeVoice-ASR teacher checkpoint into a
standalone, loadable Hugging Face Qwen2ForCausalLM.

The teacher stores the Qwen2 backbone under the key prefix
``model.language_model.*`` with a separate top-level ``lm_head.weight``. This
script streams tensors from the safetensors shards, remaps the prefix to the
plain Qwen2 layout (``model.*``), instantiates a fresh Qwen2ForCausalLM, loads
with strict=True (failing loudly on any missing/unexpected key), and saves the
result via save_pretrained.

Usage:
    python scripts/00_extract_backbone.py \
        --teacher models/teacher --out models/teacher_llm --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen2Config, Qwen2ForCausalLM

# Verified Qwen2 backbone constants (fallback if config sub-section is absent).
VERIFIED_QWEN2 = dict(
    hidden_size=3584,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
    intermediate_size=18944,
    vocab_size=152064,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=131072,
    hidden_act="silu",
    tie_word_embeddings=False,
    attention_bias=True,  # Qwen2.5 style: q/k/v have bias, o_proj none.
    use_cache=True,
)

# Sub-config keys that may hold the Qwen2 params inside the teacher config.json.
SUBCONFIG_CANDIDATES = ("decoder_config", "language_model", "llm", "text_config")

SRC_PREFIX = "model.language_model."
DST_PREFIX = "model."


def build_config(teacher_dir: Path) -> Qwen2Config:
    cfg_path = teacher_dir / "config.json"
    with cfg_path.open() as f:
        raw = json.load(f)

    sub = None
    for key in SUBCONFIG_CANDIDATES:
        cand = raw.get(key)
        if isinstance(cand, dict) and cand.get("model_type") == "qwen2":
            sub = cand
            print(f"[config] found Qwen2 sub-config under '{key}'")
            break
        if isinstance(cand, dict) and "hidden_size" in cand and "num_hidden_layers" in cand:
            sub = cand
            print(f"[config] using sub-config under '{key}' (model_type={cand.get('model_type')})")
            break

    params = dict(VERIFIED_QWEN2)
    if sub is not None:
        for k in list(params.keys()):
            if k in sub and sub[k] is not None:
                params[k] = sub[k]
        # attention_bias is not always serialized; the teacher backbone is
        # Qwen2.5 (q/k/v bias present) so keep the verified default when absent.
    else:
        print("[config] no Qwen2 sub-config found; using verified constants")

    config = Qwen2Config(**params)
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    print(
        f"[config] hidden={config.hidden_size} layers={config.num_hidden_layers} "
        f"heads={config.num_attention_heads} kv={config.num_key_value_heads} "
        f"inter={config.intermediate_size} vocab={config.vocab_size} "
        f"rope_theta={rope_theta} eps={config.rms_norm_eps} "
        f"attn_bias={config.attention_bias} tie={config.tie_word_embeddings}"
    )
    return config


def remap_key(src: str) -> str | None:
    """Map a teacher key to the standalone Qwen2 key, or None to skip."""
    if src == "lm_head.weight":
        return "lm_head.weight"
    if src.startswith(SRC_PREFIX):
        return DST_PREFIX + src[len(SRC_PREFIX):]
    return None


def load_remapped_state_dict(teacher_dir: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    index_path = teacher_dir / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    # Group keys by shard file for efficient streaming.
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    selected = 0
    for shard in sorted(shard_to_keys):
        shard_path = teacher_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for src_key in shard_to_keys[shard]:
                dst_key = remap_key(src_key)
                if dst_key is None:
                    continue
                tensor = sf.get_tensor(src_key)
                state_dict[dst_key] = tensor.to(dtype)
                selected += 1
    print(f"[load] selected {selected} backbone tensors from {len(shard_to_keys)} shards")
    return state_dict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="models/teacher", type=Path)
    ap.add_argument("--out", default="models/teacher_llm", type=Path)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    # Resolve relative to repo root (script lives in scripts/).
    repo_root = Path(__file__).resolve().parent.parent
    teacher_dir = args.teacher if args.teacher.is_absolute() else repo_root / args.teacher
    out_dir = args.out if args.out.is_absolute() else repo_root / args.out
    dtype = getattr(torch, args.dtype)

    print(f"[main] teacher={teacher_dir} out={out_dir} dtype={dtype}")

    config = build_config(teacher_dir)
    state_dict = load_remapped_state_dict(teacher_dir, dtype)

    print("[build] instantiating Qwen2ForCausalLM (meta -> materialize)...")
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
    model = model.to_empty(device="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Filter out tied-weight false positives: if lm_head is tied it may appear
    # as missing, but here tie=False so we expect a clean load.
    if missing or unexpected:
        print(f"[strict] MISSING keys ({len(missing)}):", file=sys.stderr)
        for k in missing:
            print(f"    MISSING {k}", file=sys.stderr)
        print(f"[strict] UNEXPECTED keys ({len(unexpected)}):", file=sys.stderr)
        for k in unexpected:
            print(f"    UNEXPECTED {k}", file=sys.stderr)
        raise SystemExit(
            f"FATAL: backbone did not map cleanly (missing={len(missing)}, "
            f"unexpected={len(unexpected)}). Fix the remap and retry."
        )
    print("[strict] state_dict loaded with strict match: OK (0 missing, 0 unexpected)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[params] total parameters: {n_params:,} ({n_params/1e9:.3f}B)")

    model = model.to(dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"[save] wrote standalone Qwen2ForCausalLM to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
