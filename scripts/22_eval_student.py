#!/usr/bin/env python
"""Eval trained student: generate transcripts from held-out latents, score MER
vs teacher pseudo-labels (distillation fidelity — no gold audio available)."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.eval.mer import mer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="models/student_ivod_20h")
    ap.add_argument("--manifest", default="data/pseudo/ivod_train.jsonl")
    ap.add_argument("--n", type=int, default=8, help="held-out records to eval (from the END of manifest)")
    ap.add_argument("--max-frames", type=int, default=750)
    args = ap.parse_args()

    e = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e", str(ROOT / "scripts/07e_latent_distill.py")))
    importlib.util.spec_from_file_location("e", str(ROOT / "scripts/07e_latent_distill.py")).loader.exec_module(e)
    m09 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("m", str(ROOT / "scripts/09b_audio_distill_smoke.py")))
    importlib.util.spec_from_file_location("m", str(ROOT / "scripts/09b_audio_distill_smoke.py")).loader.exec_module(m09)

    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    proc = VibeVoiceASRProcessor.from_pretrained("models/teacher", language_model_pretrained_name="Qwen/Qwen2.5-7B")
    dtype = torch.bfloat16
    student, _, _ = m09.build_student("models/teacher", "models/student_1p5b_tied_smoke", dtype)
    # load trained weights
    from safetensors.torch import load_file
    student.model.language_model.load_state_dict(load_file(f"{args.student}/language_model/model.safetensors"), strict=False)
    parts = torch.load(f"{args.student}/student_parts.pt", map_location="cpu", weights_only=False)
    student.lm_head.load_state_dict(parts["lm_head"])
    student.model.acoustic_connector.load_state_dict(parts["acoustic_connector"])
    student.model.semantic_connector.load_state_dict(parts["semantic_connector"])
    student = student.to("cuda:0").eval()
    tok = proc.tokenizer

    recs = read_manifest(str(ROOT / args.manifest))[-args.n:]  # held-out tail
    scores = []
    for r in recs:
        lp = ROOT / r.meta.get("latents_path", "")
        if not lp.exists():
            continue
        z = np.load(lp)
        ac = torch.tensor(z["acoustic"][: args.max_frames], dtype=dtype)
        se = torch.tensor(z["semantic"][: args.max_frames], dtype=dtype)
        pids, pmask = e.build_prompt(proc, ac.shape[0], "")
        ids = torch.tensor(pids).unsqueeze(0).to("cuda:0")
        mask = torch.tensor(pmask, dtype=torch.bool).unsqueeze(0).to("cuda:0")
        emb = e.splice(student, ids, mask, ac.to("cuda:0"), se.to("cuda:0"), grad=False)
        # manual greedy decode (generate() mishandles inputs_embeds-only for this model)
        gen = []
        with torch.no_grad():
            cur = emb
            for _ in range(256):
                logits = student(inputs_embeds=cur, use_cache=False).logits[:, -1, :]
                nxt = int(logits.argmax(-1))
                if nxt == tok.eos_token_id:
                    break
                gen.append(nxt)
                ne = student.get_input_embeddings()(torch.tensor([[nxt]], device="cuda:0"))
                cur = torch.cat([cur, ne], dim=1)
        hyp = tok.decode(gen, skip_special_tokens=True)
        ref = format_target(r.segments)
        s = mer(ref, hyp)
        scores.append(s)
        print(f"  {Path(r.audio_path).stem}: MER={s:.3f} | hyp[:40]={hyp[:40]!r}")
    if scores:
        print(f"\nmean MER vs teacher pseudo-label (distillation fidelity): {np.mean(scores):.3f} over {len(scores)} held-out")


if __name__ == "__main__":
    raise SystemExit(main())
