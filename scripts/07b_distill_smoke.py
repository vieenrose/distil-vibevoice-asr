#!/usr/bin/env python
"""07b_distill_smoke.py — SMOKE validation of the real distillation stack.

Drives the REAL distil_vibevoice.distill.trainer.DistillTrainer end-to-end on
the two RTX 5090s with the real prune-and-distill weights:

    teacher = models/teacher_llm       (~7.6B Qwen2)  -> cuda:1, bf16, eval, frozen
    student = models/student_1p5b_smoke(~1.5B Qwen2)  -> cuda:0, bf16, AdamW(lr=1e-4)

THROWAWAY / PLUMBING ONLY: the calibration+distill data here are synthetic
random token ids, so the resulting student is garbage. Everything is written
with a '_smoke' suffix and is not a quality run.

Data: ~40 pre-collated batches (batch=1, seqlen=512) of random input_ids with
labels = input_ids. The trainer applies the causal shift and calls the real
distill_loss (T=2.0, w_kl .5, w_ce .3, w_hidden .2) with default_layer_map(28,28)
and per-pair Linear(3584->1536) hidden projections on cuda:0. No bitsandbytes.
"""

from __future__ import annotations

import time
import json
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM

from distil_vibevoice.distill.trainer import DistillTrainer

REPO = Path("/home/luigi/distil-vibevoice-asr")
TEACHER_DIR = REPO / "models" / "teacher_llm"
STUDENT_DIR = REPO / "models" / "student_1p5b_smoke"
OUT_DIR = REPO / "models" / "distill_smoke_out"

STUDENT_DEVICE = "cuda:0"
TEACHER_DEVICE = "cuda:1"

N_BATCHES = 40
MAX_STEPS = 36
SEQLEN = 512
VOCAB = 152064


def make_batches(n: int, seqlen: int, vocab: int, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (1, seqlen), generator=g, dtype=torch.long)
        batches.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones(1, seqlen, dtype=torch.long),
                "labels": ids.clone(),
            }
        )
    return batches


def main() -> dict:
    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    seqlen = SEQLEN

    print(f"[smoke] loading teacher {TEACHER_DIR} (bf16) -> {TEACHER_DEVICE}")
    teacher = Qwen2ForCausalLM.from_pretrained(TEACHER_DIR, dtype=torch.bfloat16)
    print(f"[smoke] loading student {STUDENT_DIR} (bf16) -> {STUDENT_DEVICE}")
    student = Qwen2ForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.bfloat16)
    print(
        f"[smoke] teacher params={sum(p.numel() for p in teacher.parameters())/1e9:.2f}B "
        f"student params={sum(p.numel() for p in student.parameters())/1e9:.2f}B"
    )

    def run(seqlen: int) -> dict:
        batches = make_batches(N_BATCHES, seqlen, VOCAB)
        cfg = {
            "out_dir": str(OUT_DIR),
            "lr": 1e-4,
            "warmup_steps": 5,
            "max_steps": MAX_STEPS,
            "grad_accum": 1,
            "T": 2.0,
            "w_kl": 0.5,
            "w_ce": 0.3,
            "w_hidden": 0.2,
            "log_every": 1,
            "save_every": MAX_STEPS,  # single checkpoint at the end
            "teacher_device": TEACHER_DEVICE,
            "student_device": STUDENT_DEVICE,
            "bf16": True,
            "use_8bit_optim": False,
            "hidden_layer_map": "auto",
            "seed": 0,
        }
        trainer = DistillTrainer(student, teacher, batches, cfg)
        t0 = time.time()
        trainer.train()
        wall = time.time() - t0
        return {"trainer": trainer, "wall": wall}

    oom = False
    try:
        res = run(seqlen)
    except torch.cuda.OutOfMemoryError:
        oom = True
        print("[smoke] OOM at seqlen=512; retrying at seqlen=256")
        torch.cuda.empty_cache()
        seqlen = 256
        # fresh student to avoid a half-updated state
        student = Qwen2ForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.bfloat16)
        res = run(seqlen)

    trainer = res["trainer"]
    wall = res["wall"]

    history = trainer.history
    losses = [h["loss"] for h in history]
    finite = all(
        __import__("math").isfinite(v)
        for h in history
        for v in (h["loss"], h["kl"], h["ce"], h["hidden"])
    )
    first5 = losses[:5]
    last5 = losses[-5:]
    first5_mean = sum(first5) / len(first5)
    last5_mean = sum(last5) / len(last5)
    steps = len(losses)
    sec_per_step = wall / max(steps, 1)

    peak0 = torch.cuda.max_memory_allocated(0) / (1024**3)
    peak1 = torch.cuda.max_memory_allocated(1) / (1024**3)

    # Checkpoint roundtrip: export_hf() wrote save_pretrained() into OUT_DIR.
    ckpt_ok = False
    try:
        reloaded = Qwen2ForCausalLM.from_pretrained(OUT_DIR, dtype=torch.bfloat16)
        # compare a couple of tensors against the live student state_dict
        live = trainer.student.state_dict()
        rel = reloaded.state_dict()
        keys = list(live.keys())[:3] + list(live.keys())[-3:]
        ckpt_ok = all(
            torch.equal(live[k].cpu(), rel[k].cpu()) for k in keys if k in rel
        )
        del reloaded
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] ckpt reload failed: {e}")

    result = {
        "steps": steps,
        "loss_first5_mean": round(first5_mean, 6),
        "loss_last5_mean": round(last5_mean, 6),
        "finite": bool(finite),
        "oom": bool(oom),
        "peak_gib_cuda0": round(peak0, 3),
        "peak_gib_cuda1": round(peak1, 3),
        "ckpt_roundtrip_ok": bool(ckpt_ok),
        "sec_per_step": round(sec_per_step, 4),
        "seqlen": seqlen,
    }
    print("[smoke] RESULT " + json.dumps(result))
    return result


if __name__ == "__main__":
    main()
