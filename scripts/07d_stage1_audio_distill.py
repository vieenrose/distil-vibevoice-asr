#!/usr/bin/env python
"""Stage-1 audio-conditioned distillation with TRAINABLE connectors (real path).

Assembles the audio student (full VibeVoice-ASR + pruned/tied LLM + reprojected
connectors), enables grad on the connectors (distill.audio_encode), and trains
LLM + lm_head + tied embeddings + connector reprojection end-to-end against the
full 8.7B teacher on real pseudo-labeled meeting audio.

This is the real stage-1 loop (scaled down by --steps / --manifest for validation);
the smoke check proves connector gradient flows through the WHOLE forward+distill
loss+backward, not just an isolated encode.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.distill.audio_encode import enable_connector_training
from distil_vibevoice.distill.collator import DistillCollator
from distil_vibevoice.distill.losses import default_layer_map, distill_loss

ROOT = Path(__file__).resolve().parents[1]


def _load_09b():
    spec = importlib.util.spec_from_file_location("s09b", str(ROOT / "scripts/09b_audio_distill_smoke.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_example(processor, rec, collator, max_sec):
    m = _load_09b()
    wav = m.load_audio(rec.audio_path, max_sec=max_sec)
    hotwords = "\n".join(dict.fromkeys(s.text for s in rec.segments if s.text))[:512]
    prompt = processor(audio=[wav], context_info=hotwords, return_tensors="pt", add_generation_prompt=True)
    prompt_ids = prompt["input_ids"][0].tolist()
    prompt_acmask = prompt["acoustic_input_mask"][0].tolist()
    tok = processor.tokenizer
    tgt = tok(format_target(rec.segments), add_special_tokens=False)["input_ids"]
    tail = tgt + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    full_ids = prompt_ids + tail
    full_labels = [-100] * len(prompt_ids) + tail
    batch = collator([{"input_ids": full_ids, "labels": full_labels}])
    S = batch["input_ids"].shape[1]
    acmask = torch.zeros(1, S, dtype=torch.bool)
    acmask[0, : len(prompt_acmask)] = torch.tensor(prompt_acmask, dtype=torch.bool)
    return batch, prompt["speech_tensors"], prompt["speech_masks"], acmask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/teacher")
    ap.add_argument("--student-llm", default="models/student_1p5b_tied_smoke")
    ap.add_argument("--manifest", default="data/pseudo/simulated_relabel_manifest.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (0=only at end)")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument("--out", default="models/stage1_audio_smoke")
    args = ap.parse_args()

    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    m09 = _load_09b()
    dtype = torch.bfloat16
    dev_s, dev_t = "cuda:0", "cuda:1"

    processor = VibeVoiceASRProcessor.from_pretrained(args.teacher, language_model_pretrained_name=args.tokenizer)
    collator = DistillCollator(processor.tokenizer, max_len=16384)
    records = read_manifest(str(ROOT / args.manifest))
    print(f"{len(records)} records; training {args.steps} steps")

    teacher = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.teacher, dtype=dtype, attn_implementation="sdpa").to(dev_t).eval()
    teacher.requires_grad_(False)

    student, method, hid = m09.build_student(args.teacher, args.student_llm, dtype)
    student = student.to(dev_s)
    conn_params = enable_connector_training(student)  # <-- the fix: connectors trainable
    # trainable set: LLM + lm_head + tied embeds (already require grad) + connectors
    llm_params = [p for p in student.model.language_model.parameters() if p.requires_grad]
    head_params = [p for p in student.lm_head.parameters() if p.requires_grad]
    trainable = list({id(p): p for p in (llm_params + head_params + conn_params)}.values())
    n_conn = sum(p.numel() for p in conn_params)
    n_all = sum(p.numel() for p in trainable)
    print(f"connector method: {method}; trainable {n_all/1e6:.0f}M params incl {n_conn/1e6:.2f}M connector")

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    layer_map = default_layer_map(28, 28)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    def save_ckpt(step: int):
        d = Path(ROOT / args.out) / f"checkpoint_{step}"
        d.mkdir(parents=True, exist_ok=True)
        # student LLM (standard Qwen2) + connectors + optimizer + step
        student.model.language_model.save_pretrained(d / "language_model")
        torch.save({
            "lm_head": student.lm_head.state_dict(),
            "acoustic_connector": student.model.acoustic_connector.state_dict(),
            "semantic_connector": student.model.semantic_connector.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
        }, d / "trainer_state.pt")
        (d / "meta.json").write_text(json.dumps({"step": step, "connector_method": method}), encoding="utf-8")
        print(f"  saved checkpoint -> {d}")

    start_step = 0
    if args.resume:
        rd = Path(args.resume)
        st = torch.load(rd / "trainer_state.pt", map_location=dev_s, weights_only=False)
        from transformers import Qwen2Model  # noqa: F401
        student.model.language_model.load_state_dict(
            __import__("safetensors.torch", fromlist=["load_file"]).load_file(
                str(rd / "language_model" / "model.safetensors")), strict=False)
        student.lm_head.load_state_dict(st["lm_head"])
        student.model.acoustic_connector.load_state_dict(st["acoustic_connector"])
        student.model.semantic_connector.load_state_dict(st["semantic_connector"])
        opt.load_state_dict(st["optimizer"])
        start_step = int(st["step"]) + 1
        print(f"resumed from {args.resume} at step {start_step}")

    def fwd(model, dev, batch, st, sm, am, want_hidden):
        return model(
            input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
            speech_tensors=st.to(dev), speech_masks=sm.to(dev), acoustic_input_mask=am.to(dev),
            use_cache=False, output_hidden_states=want_hidden)

    losses, conn_grad_seen = [], False
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        rec = records[step % len(records)]
        batch, st, sm, am = build_example(processor, rec, collator, args.max_sec)
        with torch.no_grad():
            t_out = fwd(teacher, dev_t, batch, st, sm, am, False)
        s_out = fwd(student, dev_s, batch, st, sm, am, False)
        labels = batch["labels"].to(dev_s)
        tw = batch["token_weights"].to(dev_s) if "token_weights" in batch else None
        out = distill_loss(
            s_out.logits[:, :-1], t_out.logits[:, :-1].detach().to(dev_s),
            labels[:, 1:], token_weights=(tw[:, 1:] if tw is not None else None),
            T=2.0, w_kl=0.5, w_ce=0.3, w_hidden=0.0)
        opt.zero_grad()
        out["loss"].backward()
        # decisive check: connector grad flows through the FULL loss
        g = student.model.acoustic_connector.fc1.weight.grad
        if g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0:
            conn_grad_seen = True
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(out["loss"].detach().item())
        print(f"  step {step}: loss={losses[-1]:.3f} kl={float(out['kl']):.3f} ce={float(out['ce']):.3f} "
              f"lr={lr_at(step):.2e} conn_grad={'yes' if conn_grad_seen else 'no'}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(step)

    save_ckpt(args.steps - 1)
    print(f"\nloss {losses[0]:.3f} -> {losses[-1]:.3f} | connector-grad-through-full-loss: {conn_grad_seen}")
    print("(smoke: student throwaway; proves the real stage-1 loop trains connectors end-to-end)")
    return 0 if conn_grad_seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
