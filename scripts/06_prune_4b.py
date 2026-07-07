#!/usr/bin/env python
"""Stage-1 width pruning: VibeVoice-ASR 8B teacher LLM -> ~4B init.

Backbone extraction: the VibeVoice-ASR checkpoint stores its Qwen2.5-7B
decoder under a key prefix inside the safetensors shards (e.g.
``model.language_model.``). We (1) read the LLM sub-config from the teacher
config.json, (2) build a bare ``Qwen2ForCausalLM``, (3) stream matching
tensors out of the shards by suffix-remapping — never instantiating the full
multimodal model (the acoustic/semantic encoders and diffusion head stay on
disk, untouched/frozen). The extracted LLM is cached at models/teacher_llm
for reuse by 07/09. Then: compute_importance on calib batches ->
prune_qwen2_width -> save pruned init + hidden_keep_idx.pt (for
prune_connector on the audio->LLM projectors).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.pruning.importance import compute_importance
from distil_vibevoice.pruning.prune import prune_qwen2_width

LLM_CFG_KEYS = ("decoder_config", "language_model_config", "llm_config", "text_config")


def find_llm_config(teacher_dir: Path) -> dict:
    cfg = json.loads((teacher_dir / "config.json").read_text())
    for key in LLM_CFG_KEYS:
        if isinstance(cfg.get(key), dict) and "hidden_size" in cfg[key]:
            return cfg[key]
    if "hidden_size" in cfg:  # flat config
        return cfg
    raise KeyError(f"no LLM sub-config in {teacher_dir}/config.json (tried {LLM_CFG_KEYS})")


def extract_teacher_llm(teacher_dir: Path, cache_dir: Path) -> "object":
    """Extract the Qwen2 decoder from the VibeVoice checkpoint (cached)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    if (cache_dir / "config.json").exists():
        print(f"loading cached extracted LLM from {cache_dir}")
        return Qwen2ForCausalLM.from_pretrained(cache_dir, dtype=torch.bfloat16)
    from safetensors import safe_open
    index = json.loads((teacher_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    # Detect the prefix in front of the Qwen2Model keys ("...embed_tokens.weight").
    anchor = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    prefix = anchor[: -len("embed_tokens.weight")]  # e.g. "model.language_model."
    print(f"detected LLM key prefix: '{prefix}'")
    llm = Qwen2ForCausalLM(Qwen2Config(**find_llm_config(teacher_dir)))
    state: dict = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            with safe_open(teacher_dir / shard, framework="pt") as f:
                state["model." + key[len(prefix):]] = f.get_tensor(key)
        elif key.endswith("lm_head.weight"):
            with safe_open(teacher_dir / shard, framework="pt") as f:
                state["lm_head.weight"] = f.get_tensor(key)
    if "lm_head.weight" not in state:  # tied embeddings
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = llm.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"extraction mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    llm = llm.to(torch.bfloat16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm.save_pretrained(cache_dir)
    print(f"cached extracted LLM -> {cache_dir}")
    return llm


def calib_loader(manifest: Path, tokenizer_path: str, batch_size: int, seq_len: int):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    texts = [format_target(r.segments) for r in read_manifest(manifest)]

    def collate(batch: list[str]) -> dict:
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    return DataLoader(texts, batch_size=batch_size, collate_fn=collate, shuffle=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/prune_4b.yaml"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B",
                    help="ASR repo ships no tokenizer; it is pulled from Qwen2.5-7B")
    args = ap.parse_args()
    import torch
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    tgt, imp = cfg["targets"], cfg["importance"]
    teacher_dir = ROOT / cfg["model"]["source"]
    out_dir = ROOT / cfg["model"]["out_dir"]

    model = extract_teacher_llm(teacher_dir, ROOT / "models/teacher_llm")
    loader = calib_loader(ROOT / imp["calib_manifest"], args.tokenizer,
                          int(imp.get("batch_size", 2)), int(imp.get("seq_len", 4096)))
    scores = compute_importance(model, loader, num_batches=int(imp.get("calib_batches", 64)),
                                device=imp.get("device", "cuda:0"))
    pruned = prune_qwen2_width(model, scores,
                               target_hidden=int(tgt["hidden"]),
                               target_intermediate=int(tgt["intermediate"]),
                               target_q_heads=int(tgt["q_heads"]),
                               target_kv_heads=int(tgt["kv_heads"]),
                               tie_word_embeddings=bool(
                                   cfg.get("embeddings", {}).get("tie_word_embeddings", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    pruned.save_pretrained(out_dir)
    # Channel index set for prune_connector on the audio->LLM projectors
    # (same top-k rule prune_qwen2_width applies to hidden channels).
    keep_idx = torch.topk(scores.hidden, int(tgt["hidden"])).indices.sort().values
    torch.save(keep_idx, out_dir / "hidden_keep_idx.pt")
    # TODO(audio): apply prune_connector to the acoustic/semantic connectors once
    # the connector modules are wired in (text-only distillation runs without them).
    n = sum(p.numel() for p in pruned.parameters()) / 1e9
    print(f"pruned model: {n:.2f}B params -> {out_dir} (+ hidden_keep_idx.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
