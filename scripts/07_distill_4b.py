#!/usr/bin/env python
"""Stage-1 distillation: 8B teacher -> pruned 4B student (text-only milestone).

Builds a dataloader over the train manifest — each record's segments are
rendered with format_target() and tokenized with the Qwen2.5 tokenizer —
then runs DistillTrainer (KL + CE + hidden-MSE, speaker/timestamp tokens
upweighted 4x). The teacher is the Qwen2 backbone extracted/cached by
scripts/06_prune_4b.py at models/teacher_llm.

TODO(audio): splice frozen acoustic+semantic encoder latents into the input
('audio_latents' feature) so the student conditions on speech, not only on
target text. Text-only distillation runs end-to-end as the first milestone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.distill.collator import DistillCollator
from distil_vibevoice.distill.trainer import DistillTrainer


def build_loader(manifest: Path, tokenizer, cfg: dict):
    from torch.utils.data import DataLoader, Dataset

    max_len = max((c["max_len"] for c in (cfg.get("data") or {}).get("seq_len_curriculum", [])),
                  default=16384)
    records = read_manifest(manifest)

    class TargetTextDataset(Dataset):
        """Tokenized format_target(segments); audio latents TODO (see module doc)."""

        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, i: int) -> dict:
            ids = tokenizer(format_target(records[i].segments), truncation=True,
                            max_length=max_len).input_ids
            return {"input_ids": ids, "labels": list(ids)}
            # TODO(audio): add 'audio_latents' from the frozen encoders here.

    collator = DistillCollator(tokenizer, max_len=max_len,
                               speaker_ts_upweight=float(
                                   (cfg.get("loss") or {}).get("speaker_ts_upweight", 4.0)))
    bs = int((cfg.get("train") or {}).get("micro_batch_size", 1))
    return DataLoader(TargetTextDataset(), batch_size=bs, shuffle=True,
                      collate_fn=collator, num_workers=2, drop_last=True)


def trainer_cfg(cfg: dict, out_dir: Path) -> dict:
    """Flatten loss/optim/train sections (nested kept too) for DistillTrainer."""
    flat: dict = {"out_dir": str(out_dir)}
    for section in ("loss", "optim", "train", "data", "model"):
        flat[section] = cfg.get(section) or {}
        flat.update(flat[section])
    return flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/distill_stage1_4b.yaml"))
    ap.add_argument("--teacher-llm", default=str(ROOT / "models/teacher_llm"),
                    help="Qwen2 backbone extracted by 06_prune_4b.py")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    args = ap.parse_args()
    import torch
    import yaml
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    cfg = yaml.safe_load(Path(args.config).read_text())
    mdl = cfg["model"]
    out_dir = ROOT / mdl["out_dir"]
    teacher_dir = Path(args.teacher_llm)
    if not (teacher_dir / "config.json").exists():
        sys.exit(f"extracted teacher LLM not found at {teacher_dir} — run scripts/06_prune_4b.py first")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    student = Qwen2ForCausalLM.from_pretrained(ROOT / mdl["student_init"], dtype=torch.bfloat16)
    teacher = Qwen2ForCausalLM.from_pretrained(teacher_dir, dtype=torch.bfloat16)

    loader = build_loader(ROOT / (cfg.get("data") or {})["train_manifest"], tokenizer, cfg)
    tcfg = trainer_cfg(cfg, out_dir)
    tcfg.setdefault("student_device", mdl.get("student_device", "cuda:0"))
    tcfg.setdefault("teacher_device", mdl.get("teacher_device", "cuda:1"))
    tcfg["tokenizer"] = tokenizer

    print(f"stage-1 distillation: student {mdl['student_init']} <- teacher {teacher_dir}")
    DistillTrainer(student, teacher, loader, tcfg).train()
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
