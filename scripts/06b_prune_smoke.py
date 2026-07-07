#!/usr/bin/env python
"""SMOKE plumbing validation of the real Minitron width-prune on GPU.

Takes the standalone extracted backbone at models/teacher_llm (a real
Qwen2ForCausalLM, ~7.6B params in bf16) and runs the full width-prune path to
the stage-2 1.5B geometry, purely to validate the plumbing on the REAL weights.

IMPORTANT: this is NOT a quality run. The calibration data is synthetic random
token ids, so importance scores are meaningless and the resulting student is a
throwaway. Output is named with a '_smoke' suffix to make that explicit.

Pipeline:
  1. Load models/teacher_llm on cuda:0 in bf16, eval().
  2. Tiny synthetic calibration loader: 16 batches of random input_ids
     (batch 2, seqlen 256, vocab 152064) as {'input_ids','attention_mask'}.
  3. compute_importance(..., num_batches=16, device='cuda:0').
  4. prune_qwen2_width -> student at hidden 1536 / intermediate 8960 /
     12 Q heads / 2 KV heads (head_dim stays 128 -> 12*128 = 1536).
  5. Verify finite logits [B,T,152064], print param counts, assert smaller.
  6. save_pretrained -> models/student_1p5b_smoke; report peak cuda:0 GiB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from distil_vibevoice.pruning.importance import compute_importance
from distil_vibevoice.pruning.prune import prune_qwen2_width

DEVICE = "cuda:0"
VOCAB = 152064
CALIB_BATCHES = 16
BATCH = 2
SEQLEN = 256

# Stage-2 1.5B targets (head_dim stays 128, so 12*128 = 1536 hidden).
TGT_HIDDEN = 1536
TGT_INTERMEDIATE = 8960
TGT_Q_HEADS = 12
TGT_KV_HEADS = 2


def synthetic_calib_loader():
    """16 batches of random token ids as {'input_ids','attention_mask'} on cuda:0.

    Random calibration = plumbing only; importance scores are meaningless.
    """
    g = torch.Generator().manual_seed(0)
    for _ in range(CALIB_BATCHES):
        ids = torch.randint(0, VOCAB, (BATCH, SEQLEN), generator=g)
        yield {
            "input_ids": ids.to(DEVICE),
            "attention_mask": torch.ones(BATCH, SEQLEN, dtype=torch.long, device=DEVICE),
        }


def main() -> int:
    from transformers import Qwen2ForCausalLM

    teacher_dir = ROOT / "models/teacher_llm"
    out_dir = ROOT / "models/student_1p5b_smoke"

    print(f"[smoke] SYNTHETIC random calibration -- throwaway student, plumbing only")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    print(f"[smoke] loading {teacher_dir} on {DEVICE} (bf16, eval)")
    model = Qwen2ForCausalLM.from_pretrained(teacher_dir, dtype=torch.bfloat16)
    model = model.to(DEVICE).eval()
    teacher_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] teacher params: {teacher_params/1e9:.3f}B")

    print(f"[smoke] compute_importance over {CALIB_BATCHES} random batches")
    scores = compute_importance(model, synthetic_calib_loader(),
                                num_batches=CALIB_BATCHES, device=DEVICE)

    print(f"[smoke] prune_qwen2_width -> hidden={TGT_HIDDEN} inter={TGT_INTERMEDIATE} "
          f"q={TGT_Q_HEADS} kv={TGT_KV_HEADS}")
    student = prune_qwen2_width(
        model, scores,
        target_hidden=TGT_HIDDEN,
        target_intermediate=TGT_INTERMEDIATE,
        target_q_heads=TGT_Q_HEADS,
        target_kv_heads=TGT_KV_HEADS,
    )
    student_params = sum(p.numel() for p in student.parameters())
    print(f"[smoke] student params: {student_params/1e9:.3f}B")
    assert student_params < teacher_params, "student not smaller than teacher!"
    print(f"[smoke] student is {teacher_params/student_params:.2f}x smaller than teacher")

    # Free the teacher before the verification forward to keep cuda:0 headroom.
    del model
    torch.cuda.empty_cache()

    print(f"[smoke] verification forward pass on random batch")
    student = student.to(DEVICE).eval()
    ids = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    with torch.no_grad():
        logits = student(input_ids=ids, use_cache=False).logits
    expected = (BATCH, SEQLEN, VOCAB)
    shape_ok = tuple(logits.shape) == expected
    finite_ok = bool(torch.isfinite(logits).all().item())
    forward_ok = shape_ok and finite_ok
    print(f"[smoke] logits shape={tuple(logits.shape)} expected={expected} "
          f"finite={finite_ok} -> forward_ok={forward_ok}")

    saved_ok = False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(out_dir)
        saved_ok = (out_dir / "config.json").exists() and any(out_dir.glob("*.safetensors"))
        print(f"[smoke] saved student -> {out_dir} (saved_ok={saved_ok})")
    except Exception as e:  # pragma: no cover - report but don't crash the smoke
        print(f"[smoke] save failed: {e!r}")

    peak_gib = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    print(f"[smoke] peak cuda:0 allocated: {peak_gib:.2f} GiB")

    result = {
        "param_count_b": round(student_params / 1e9, 4),
        "teacher_param_count_b": round(teacher_params / 1e9, 4),
        "forward_ok": forward_ok,
        "saved_ok": bool(saved_ok),
        "peak_gib": round(peak_gib, 2),
    }
    print("[smoke] RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
