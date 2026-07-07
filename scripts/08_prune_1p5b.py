#!/usr/bin/env python
"""Stage-2 width pruning: distilled 4B student -> ~1.5B init.

Same Minitron-style recipe as 06 but the source is already a plain
Qwen2ForCausalLM checkpoint (checkpoints/distill_4b), so no backbone
extraction is needed: compute_importance on calib batches ->
prune_qwen2_width to the 1.5B targets (hidden 1536, intermediate 8960,
12 Q heads, 2 KV heads) -> save init + hidden_keep_idx.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.pruning.importance import compute_importance
from distil_vibevoice.pruning.prune import prune_qwen2_width


def calib_loader(manifest: Path, tokenizer_path: str, batch_size: int, seq_len: int):
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
    ap.add_argument("--config", default=str(ROOT / "configs/prune_1p5b.yaml"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    args = ap.parse_args()
    import torch
    import yaml
    from transformers import Qwen2ForCausalLM

    cfg = yaml.safe_load(Path(args.config).read_text())
    tgt, imp = cfg["targets"], cfg["importance"]
    src_dir = ROOT / cfg["model"]["source"]
    out_dir = ROOT / cfg["model"]["out_dir"]

    print(f"loading 4B student from {src_dir}")
    model = Qwen2ForCausalLM.from_pretrained(src_dir, dtype=torch.bfloat16)
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
    keep_idx = torch.topk(scores.hidden, int(tgt["hidden"])).indices.sort().values
    torch.save(keep_idx, out_dir / "hidden_keep_idx.pt")
    # TODO(audio): prune_connector on audio->LLM projectors with hidden_keep_idx.
    n = sum(p.numel() for p in pruned.parameters()) / 1e9
    print(f"pruned model: {n:.2f}B params -> {out_dir} (+ hidden_keep_idx.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
