#!/usr/bin/env python
"""Stage-2 distillation: 4B (TA) teacher -> pruned 1.5B student.

Like stage 1, plus: the original 8B teacher (extracted Qwen2 backbone from
06) is loaded and wired into the trainer cfg as ``teacher_8b`` with
``direct_8b_fraction`` (default 0.1) — that fraction of batches is distilled
directly from the 8B instead of the 4B TA. Length/speaker/seq-len curricula
are passed through in cfg for the trainer to schedule.

TODO(audio): 'audio_latents' hookup, same milestone note as 07.
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
    flat: dict = {"out_dir": str(out_dir)}
    for section in ("loss", "optim", "train", "data", "model"):
        flat[section] = cfg.get(section) or {}
        flat.update(flat[section])
    return flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/distill_stage2_1p5b.yaml"))
    ap.add_argument("--teacher-8b-llm", default=str(ROOT / "models/teacher_llm"),
                    help="8B Qwen2 backbone extracted by 06_prune_4b.py")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--no-direct-8b", action="store_true",
                    help="skip loading the 8B (TA-only distillation)")
    args = ap.parse_args()
    import torch
    import yaml
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    cfg = yaml.safe_load(Path(args.config).read_text())
    mdl = cfg["model"]
    out_dir = ROOT / mdl["out_dir"]
    ta_path = ROOT / mdl.get("ta_teacher", mdl["teacher"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    student = Qwen2ForCausalLM.from_pretrained(ROOT / mdl["student_init"], dtype=torch.bfloat16)
    ta_teacher = Qwen2ForCausalLM.from_pretrained(ta_path, dtype=torch.bfloat16)

    loader = build_loader(ROOT / (cfg.get("data") or {})["train_manifest"], tokenizer, cfg)
    tcfg = trainer_cfg(cfg, out_dir)
    tcfg.setdefault("student_device", mdl.get("student_device", "cuda:0"))
    tcfg.setdefault("teacher_device", mdl.get("teacher_device", "cuda:1"))
    tcfg["tokenizer"] = tokenizer
    tcfg["direct_8b_fraction"] = 0.0 if args.no_direct_8b \
        else float(mdl.get("direct_8b_fraction", 0.1))
    if tcfg["direct_8b_fraction"] > 0:
        t8 = Path(args.teacher_8b_llm)
        if not (t8 / "config.json").exists():
            sys.exit(f"8B backbone not found at {t8} — run scripts/06_prune_4b.py "
                     "first, or pass --no-direct-8b")
        # Wired for the trainer: kept on teacher_device alongside the 4B TA
        # (both fit in bf16 on one RTX 5090 only if offloaded — trainer decides).
        tcfg["teacher_8b"] = Qwen2ForCausalLM.from_pretrained(t8, dtype=torch.bfloat16)
        print(f"direct-8B distillation on {tcfg['direct_8b_fraction']:.0%} of batches")

    print(f"stage-2 distillation: student {mdl['student_init']} <- TA {ta_path}")
    DistillTrainer(student, ta_teacher, loader, tcfg).train()
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
