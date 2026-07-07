#!/usr/bin/env python
"""Audio-free latent-based stage-1 distillation.

Trains on cached σ-VAE latents (no raw audio needed). Both teacher and student
consume the same cached acoustic (64d) + semantic (128d) latents through their
OWN connectors, so:
  - teacher connectors (64/128 -> 3584, frozen) produce the soft-target logits,
  - student connectors (64/128 -> 1536, TRAINABLE) + pruned LLM produce the student
    logits,
and distill_loss trains the student end-to-end. The audio-token prompt layout is
reconstructed from the latent frame count (= processor vae_tok_len), so we never
touch a wav. This is what lets a 1000-hour corpus train from ~6 GB of latents.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from distil_vibevoice.data.manifest import format_target, read_manifest
from distil_vibevoice.distill.audio_encode import enable_connector_training
from distil_vibevoice.distill.losses import distill_loss

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]


def build_prompt(processor, n_frames: int, context_info: str = "") -> tuple[list[int], list[int]]:
    """Reconstruct (input_ids, acoustic_mask) for n_frames audio tokens — no audio."""
    tok = processor.tokenizer
    sid, pid, eid = processor.speech_start_id, processor.speech_pad_id, processor.speech_end_id
    dur = n_frames * 3200 / 24000.0
    sys_txt = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}], tokenize=False)
    sys_tokens = tok.encode(sys_txt)
    st, pt, et = (tok.convert_ids_to_tokens(x) for x in (sid, pid, eid))
    if context_info.strip():
        suffix = f"This is a {dur:.2f} seconds audio, with extra info: {context_info.strip()}\n\nPlease transcribe it with these keys: " + ", ".join(SHOW_KEYS)
    else:
        suffix = f"This is a {dur:.2f} seconds audio, please transcribe it with these keys: " + ", ".join(SHOW_KEYS)
    user_str = "".join([st] + [pt] * n_frames + [et]) + "\n" + suffix
    user_tokens = tok.apply_chat_template([{"role": "user", "content": user_str}], tokenize=True)
    full = sys_tokens + user_tokens
    mask = [1 if t == pid else 0 for t in full]
    return full, mask


def splice(model, input_ids, acoustic_mask, ac_lat, se_lat, grad: bool):
    """inputs_embeds with connector(latents) spliced at audio positions."""
    emb = model.get_input_embeddings()(input_ids)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        feats = model.model.acoustic_connector(ac_lat) + model.model.semantic_connector(se_lat)
    emb = emb.clone()
    emb[acoustic_mask] = feats.to(emb.dtype)
    return emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/teacher")
    ap.add_argument("--student-llm", default="models/student_1p5b_tied_smoke")
    ap.add_argument("--manifest", default="data/pseudo/ivod_stream_manifest.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-frames", type=int, default=1800, help="cap audio frames/example (VRAM); 1800=4min")
    ap.add_argument("--no-teacher", action="store_true",
                    help="GPU0-only: skip the teacher, train CE on pseudo-labels (sequence-level/hard-label distillation). "
                         "Fits one GPU; soft-label KD needs 2 GPUs or cached teacher logits.")
    ap.add_argument("--out", default="")
    ap.add_argument("--sanity", action="store_true", help="CPU: verify prompt reconstruction vs processor, no training")
    args = ap.parse_args()

    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    processor = VibeVoiceASRProcessor.from_pretrained(args.teacher, language_model_pretrained_name=args.tokenizer)

    if args.sanity:
        # verify our reconstruction matches the processor for a dummy audio
        n = 37
        ref = processor(audio=[np.zeros(n * 3200, dtype=np.float32)], return_tensors="pt", add_generation_prompt=True)
        ours_ids, ours_mask = build_prompt(processor, n)
        ref_ids = ref["input_ids"][0].tolist()
        ref_mask = ref["acoustic_input_mask"][0].tolist()
        ok = ref_ids == ours_ids and ref_mask == ours_mask
        print(f"sanity: reconstructed prompt matches processor = {ok} (len {len(ours_ids)} vs {len(ref_ids)}, "
              f"audio_positions {sum(ours_mask)} vs {sum(ref_mask)})")
        return 0 if ok else 1

    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    m09 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("s09b", str(ROOT / "scripts/09b_audio_distill_smoke.py")))
    importlib.util.spec_from_file_location("s09b", str(ROOT / "scripts/09b_audio_distill_smoke.py")).loader.exec_module(m09)

    dtype = torch.bfloat16
    dev_s = "cuda:0"
    teacher = None
    if not args.no_teacher:
        dev_t = "cuda:1"
        teacher = VibeVoiceASRForConditionalGeneration.from_pretrained(args.teacher, dtype=dtype, attn_implementation="sdpa").to(dev_t).eval()
        teacher.requires_grad_(False)
    student, method, hid = m09.build_student(args.teacher, args.student_llm, dtype)
    student = student.to(dev_s)
    # gradient checkpointing on the student LLM: trades compute for a big activation-memory cut (fits 1 GPU)
    try:
        student.model.language_model.gradient_checkpointing_enable()
        student.model.language_model.config.use_cache = False
    except Exception as e:
        print(f"grad-checkpoint enable skipped: {e}")
    conn = enable_connector_training(student)
    trainable = list({id(p): p for p in ([p for p in student.model.language_model.parameters() if p.requires_grad]
                     + [p for p in student.lm_head.parameters() if p.requires_grad] + conn)}.values())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"connector method {method}; {sum(p.numel() for p in trainable)/1e6:.0f}M trainable")

    records = [r for r in read_manifest(str(ROOT / args.manifest))]
    tok = processor.tokenizer
    losses = []
    for step in range(args.steps):
        rec = records[step % len(records)]
        latp = None
        if rec.meta.get("latents_path"):
            latp = ROOT / rec.meta["latents_path"]
        else:  # fallback: derive from audio stem
            cand = ROOT / "data/latents/ivod" / f"{Path(rec.audio_path).stem}.npz"
            if cand.exists():
                latp = cand
        if not latp or not latp.exists():
            continue
        z = np.load(latp)
        ac = torch.tensor(z["acoustic"][: args.max_frames], dtype=dtype)
        se = torch.tensor(z["semantic"][: args.max_frames], dtype=dtype)
        n = ac.shape[0]
        hot = "\n".join(dict.fromkeys(s.text for s in rec.segments if s.text and not s.text.startswith("[")))[:512]
        pids, pmask = build_prompt(processor, n, hot)
        tgt = tok(format_target(rec.segments), add_special_tokens=False)["input_ids"][:1024] + [tok.eos_token_id]
        ids = torch.tensor(pids + tgt).unsqueeze(0)
        mask = torch.tensor(pmask + [0] * len(tgt), dtype=torch.bool).unsqueeze(0)
        labels = torch.tensor([-100] * len(pids) + tgt).unsqueeze(0)

        se_emb = splice(student, ids.to(dev_s), mask.to(dev_s), ac.to(dev_s), se.to(dev_s), grad=True)
        s_logits = student(inputs_embeds=se_emb, use_cache=False).logits
        if args.no_teacher:  # GPU0-only: direct CE on pseudo-labels (sequence-level distillation)
            import torch.nn.functional as F
            lg = s_logits[:, :-1].reshape(-1, s_logits.size(-1))
            tg = labels[:, 1:].reshape(-1).to(dev_s)
            loss = F.cross_entropy(lg.float(), tg, ignore_index=-100)
            out = {"loss": loss, "kl": torch.tensor(0.0), "ce": loss.detach()}
        else:
            with torch.no_grad():
                te = splice(teacher, ids.to(dev_t), mask.to(dev_t), ac.to(dev_t), se.to(dev_t), grad=False)
                t_logits = teacher(inputs_embeds=te, use_cache=False).logits
            out = distill_loss(s_logits[:, :-1], t_logits[:, :-1].detach().to(dev_s), labels[:, 1:].to(dev_s),
                               T=2.0, w_kl=0.5, w_ce=0.3, w_hidden=0.0)
        opt.zero_grad(); out["loss"].backward()
        cg = student.model.acoustic_connector.fc1.weight.grad
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        losses.append(out["loss"].detach().item())
        print(f"  step {step}: loss={losses[-1]:.3f} frames={n} conn_grad={'yes' if cg is not None and float(cg.abs().sum())>0 else 'no'}")

    if losses:
        print(f"\nlatent-distill loss {losses[0]:.3f} -> {losses[-1]:.3f} (audio-free, from cached latents)")
    if args.out:
        d = Path(ROOT / args.out); d.mkdir(parents=True, exist_ok=True)
        student.model.language_model.save_pretrained(d / "language_model")
        torch.save({"lm_head": student.lm_head.state_dict(),
                    "acoustic_connector": student.model.acoustic_connector.state_dict(),
                    "semantic_connector": student.model.semantic_connector.state_dict(),
                    "steps": len(losses), "final_loss": losses[-1] if losses else None}, d / "student_parts.pt")
        print(f"saved trained student -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
