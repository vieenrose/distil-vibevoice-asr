#!/usr/bin/env python
"""07c_textdistill_real.py — REAL text-pathway distillation on structured meeting targets.

Unlike 07b (random-token smoke), this drives the real distillation stack on
REAL data: the Traditional-Chinese + English meeting transcripts in
``data/manifests/simulated.jsonl`` rendered to the teacher's structured output
format via ``manifest.format_target`` (JSON array of
{Start, End, Speaker, Content} segments), tokenized with the REAL Qwen2.5
152k-vocab tokenizer (``models/tokenizer``).

    teacher = models/teacher_llm            (~7.6B Qwen2)  -> cuda:1, bf16, eval, frozen
    student = models/student_1p5b_tied_smoke (~1.5B tied)  -> cuda:0, bf16, AdamW

This validates EVERYTHING on the text side — real tokenizer, DistillCollator
markup upweighting, DistillTrainer causal-shift + KL/CE/hidden loss, two-GPU
placement, checkpoint export — on real structured targets. The AUDIO encoder
pathway is NOT exercised (the vibevoice package is not installed); there is no
audio conditioning, so the student learns the target text distribution from the
teacher's logits over the same (prompt + target) sequence. The student weights
here still originate from a smoke prune, so this is a plumbing/behaviour
validation, not a quality run — but the loss should descend more cleanly on
real structured tokens than on random ids.

Each example = a fixed instruction prompt (tokens masked to -100) followed by
the structured target (labels = target token ids). Sequences up to 1024 tokens.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2ForCausalLM

from distil_vibevoice.data.manifest import read_manifest, format_target
from distil_vibevoice.distill.collator import DistillCollator
from distil_vibevoice.distill.trainer import DistillTrainer

REPO = Path("/home/luigi/distil-vibevoice-asr")
TOKENIZER_DIR = REPO / "models" / "tokenizer"
TEACHER_DIR = REPO / "models" / "teacher_llm"
STUDENT_DIR = REPO / "models" / "student_1p5b_tied_smoke"
MANIFEST = REPO / "data" / "manifests" / "simulated.jsonl"
OUT_DIR = REPO / "models" / "textdistill_real_out"

STUDENT_DEVICE = "cuda:0"
TEACHER_DEVICE = "cuda:1"

MAX_STEPS = 30
MAX_LEN = 1024
# Short fixed instruction; ASR-style, no audio available so it just conditions
# the structured-JSON output format. Prompt tokens are masked from the loss.
INSTRUCTION = (
    "Transcribe the meeting audio into a JSON array of segments with "
    "Start, End, Speaker and Content fields.\n"
)


def build_features(tokenizer, records) -> list[dict]:
    """One feature per record: prompt (masked) + structured target (labelled)."""
    prompt_ids = tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
    feats: list[dict] = []
    for rec in records:
        target = format_target(rec.segments)
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        # Append EOS so the target is a complete generation.
        eos = tokenizer.eos_token_id
        if eos is not None:
            target_ids = target_ids + [int(eos)]
        input_ids = list(prompt_ids) + list(target_ids)
        labels = [-100] * len(prompt_ids) + list(target_ids)
        feats.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return feats


def main() -> dict:
    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    torch.manual_seed(0)

    print(f"[07c] loading tokenizer {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    records = read_manifest(MANIFEST)
    print(f"[07c] manifest records: {len(records)}")
    feats = build_features(tokenizer, records)
    lens = [len(f["input_ids"]) for f in feats]
    tgt_lens = [sum(1 for x in f["labels"] if x != -100) for f in feats]
    durs = [r.duration_s for r in records]
    tok_per_s = sum(tgt_lens) / sum(durs)
    print(
        f"[07c] examples={len(feats)} seqlen[min/mean/max]="
        f"{min(lens)}/{sum(lens)//len(lens)}/{max(lens)} "
        f"mean_target_tokens/s_audio={tok_per_s:.3f}"
    )

    coll = DistillCollator(tokenizer, max_len=MAX_LEN)
    print(f"[07c] collator special_token_ids (n={len(coll.special_token_ids)}): "
          f"{ {i: tokenizer.decode([i]) for i in sorted(coll.special_token_ids)} }")

    print(f"[07c] loading teacher {TEACHER_DIR} (bf16) -> {TEACHER_DEVICE}")
    teacher = Qwen2ForCausalLM.from_pretrained(TEACHER_DIR, dtype=torch.bfloat16)
    print(f"[07c] loading student {STUDENT_DIR} (bf16) -> {STUDENT_DEVICE}")
    student = Qwen2ForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.bfloat16)
    print(
        f"[07c] teacher={sum(p.numel() for p in teacher.parameters())/1e9:.3f}B "
        f"student={sum(p.numel() for p in student.parameters())/1e9:.3f}B"
    )

    def run(max_len: int) -> DistillTrainer:
        c = DistillCollator(tokenizer, max_len=max_len)
        loader = DataLoader(feats, batch_size=1, shuffle=True, collate_fn=c)
        cfg = {
            "out_dir": str(OUT_DIR),
            "lr": 2e-4,
            "warmup_steps": 5,
            "max_steps": MAX_STEPS,
            "grad_accum": 1,
            "T": 2.0,
            "w_kl": 0.5,
            "w_ce": 0.3,
            "w_hidden": 0.2,
            "log_every": 1,
            "save_every": MAX_STEPS,
            "teacher_device": TEACHER_DEVICE,
            "student_device": STUDENT_DEVICE,
            "bf16": True,
            "use_8bit_optim": False,
            "hidden_layer_map": "auto",
            "seed": 0,
            "tokenizer": tokenizer,
        }
        trainer = DistillTrainer(student, teacher, loader, cfg)
        trainer.train()
        return trainer

    oom = False
    max_len = MAX_LEN
    try:
        trainer = run(max_len)
    except torch.cuda.OutOfMemoryError:
        oom = True
        print(f"[07c] OOM at max_len={max_len}; retrying at 512")
        torch.cuda.empty_cache()
        max_len = 512
        student = Qwen2ForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.bfloat16)
        trainer = run(max_len)

    history = trainer.history
    losses = [h["loss"] for h in history]
    finite = all(
        math.isfinite(v)
        for h in history
        for v in (h["loss"], h["kl"], h["ce"], h["hidden"])
    )
    first5 = losses[:5]
    last5 = losses[-5:]
    result = {
        "steps": len(losses),
        "loss_first5": round(sum(first5) / len(first5), 6),
        "loss_last5": round(sum(last5) / len(last5), 6),
        "finite": bool(finite),
        "oom": bool(oom),
        "peak_gib_cuda0": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
        "peak_gib_cuda1": round(torch.cuda.max_memory_allocated(1) / 1024**3, 3),
        "max_len": max_len,
        "tok_per_s_audio": round(tok_per_s, 3),
        "special_ids": sorted(coll.special_token_ids),
    }
    print("[07c] RESULT " + json.dumps(result))
    return result


if __name__ == "__main__":
    main()
