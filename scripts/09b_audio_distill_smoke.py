#!/usr/bin/env python
"""09b: assemble a pruned-LLM VibeVoice-ASR STUDENT and run a real
audio-conditioned forward + short distill smoke against the full teacher.

Pipeline
--------
1. Load the full teacher ``VibeVoiceASRForConditionalGeneration`` on cuda:1.
2. Assemble the student (on CPU, then -> cuda:0): take a second copy of the
   teacher and REPLACE ``model.language_model`` (a Qwen2 base, hidden 3584)
   with the pruned tied LLM at ``models/student_1p5b_tied_smoke`` (hidden 1536)
   and swap ``lm_head`` for the pruned 1536->152064 head. Re-project the
   acoustic + semantic ``SpeechConnector`` outputs 3584 -> 1536.

   Connector re-projection: the connectors are ``SpeechConnector`` =
   ``fc1(in->H) -> RMSNorm(H) -> fc2(H->H)``. If the pruned student's
   hidden keep-index (which teacher hidden channels survived) is RECOVERABLE
   -- we recover it by exact-matching the student's ``embed_tokens`` columns
   against the teacher's, since width-pruning copies embedding columns
   verbatim -- we slice fc1.out / norm / fc2.in+out along that index (this is
   the faithful analogue of ``pruning.prune.prune_connector``, extended to also
   slice the RMSNorm, which ``prune_connector`` alone skips because it only
   touches ``nn.Linear``). Otherwise we re-init fresh Linears to 1536.
   Which path was taken is reported in ``connector_method``.

   NOTE: ``encode_speech`` runs the encoders AND connectors under
   ``torch.no_grad()``, so the connectors are effectively frozen during the
   forward; the audio features enter the LLM as constants. The distill step
   therefore trains the student LLM + lm_head (+ tied embeddings) conditioned
   on real audio features. Encoders are frozen and share teacher weights.
3. Single forward on one real ~30s clip through BOTH models; confirm student
   logits are [1, S, 152064], match the teacher seq length, and are finite.
4. ~15 audio-conditioned distill steps with ``distill.losses.distill_loss``
   (teacher audio-logits = soft targets; real manifest transcript = CE labels
   via ``DistillCollator``; kept_vocab_ids=None).

Reports assembled/forward/distill flags, first/last loss, peak GiB per GPU,
sec/step. Honest smoke: no quality claim, tiny step count, single clip.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


# --------------------------------------------------------------------------
def load_audio(path: str, max_sec: float = 30.0, sr: int = 24000):
    import numpy as np
    import soundfile as sf

    wav, file_sr = sf.read(path, dtype="float32")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    n = int(max_sec * sr)
    return np.asarray(wav[:n], dtype="float32")


def recover_keep_idx(teacher_embed: torch.Tensor, student_embed: torch.Tensor):
    """Recover which teacher hidden columns the student kept.

    Width-pruning copies ``embed_tokens[:, keep_idx]`` verbatim, so each student
    column equals exactly one teacher column. Fingerprint columns on a few rows
    to get candidates, then verify full-column equality. Returns (idx, "exact")
    or (None, reason).
    """
    te = teacher_embed.detach().float()  # [V, H]
    se = student_embed.detach().float()  # [V, h]
    V, H = te.shape
    h = se.shape[1]
    rows = torch.linspace(0, V - 1, steps=32).long()
    tf = te[rows].t().contiguous()  # [H, 32]
    sf_ = se[rows].t().contiguous()  # [h, 32]
    d = torch.cdist(sf_, tf)  # [h, H]
    idx = d.argmin(dim=1)  # [h]
    if idx.unique().numel() != h:
        return None, "non_unique_fingerprint_match"
    full_max_diff = (te[:, idx] - se).abs().max().item()
    if full_max_diff < 1e-3:
        return idx, "exact"
    return None, f"col_mismatch_max_diff={full_max_diff:.4g}"


def slice_speech_connector(conn, keep_idx: torch.Tensor):
    """Slice a SpeechConnector (fc1->RMSNorm->fc2) output dim along keep_idx."""
    from vibevoice.modular.modeling_vibevoice import SpeechConnector

    in_dim = conn.fc1.in_features
    out_dim = int(keep_idx.numel())
    new = SpeechConnector(in_dim, out_dim)
    ki = keep_idx.to(conn.fc1.weight.device)
    with torch.no_grad():
        new.fc1.weight.copy_(conn.fc1.weight[ki])
        if conn.fc1.bias is not None:
            new.fc1.bias.copy_(conn.fc1.bias[ki])
        new.norm.weight.copy_(conn.norm.weight[ki])
        new.fc2.weight.copy_(conn.fc2.weight[ki][:, ki])
        if conn.fc2.bias is not None:
            new.fc2.bias.copy_(conn.fc2.bias[ki])
    return new.to(dtype=conn.fc1.weight.dtype)


def build_student(teacher_path: str, student_llm_path: str, dtype):
    """Assemble the pruned-LLM student VibeVoice-ASR on CPU. Returns
    (student_model, connector_method, target_hidden)."""
    from transformers import Qwen2ForCausalLM
    from vibevoice.modular.modeling_vibevoice import SpeechConnector
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )

    student = VibeVoiceASRForConditionalGeneration.from_pretrained(
        teacher_path, dtype=dtype, attn_implementation="sdpa"
    )
    teacher_embed = student.get_input_embeddings().weight  # [V, 3584]

    llm = Qwen2ForCausalLM.from_pretrained(student_llm_path, dtype=dtype)
    student_embed = llm.model.embed_tokens.weight  # [V, 1536]
    target_hidden = int(student_embed.shape[1])

    keep_idx, why = recover_keep_idx(teacher_embed, student_embed)

    old_ac = student.model.acoustic_connector
    old_se = student.model.semantic_connector
    if keep_idx is not None:
        new_ac = slice_speech_connector(old_ac, keep_idx)
        new_se = slice_speech_connector(old_se, keep_idx)
        connector_method = f"sliced_along_recovered_keep_idx({why})"
    else:
        new_ac = SpeechConnector(student.config.acoustic_vae_dim, target_hidden)
        new_se = SpeechConnector(student.config.semantic_vae_dim, target_hidden)
        connector_method = f"reinit_fresh_linear_{target_hidden}({why})"
    student.model.acoustic_connector = new_ac.to(dtype)
    student.model.semantic_connector = new_se.to(dtype)

    # Swap the language model (Qwen2 base) and the lm_head.
    student.model.language_model = llm.model
    student.lm_head = llm.lm_head
    student.vocab_size = int(llm.config.vocab_size)

    # Keep config.decoder_config consistent (metadata for a later save).
    dc = student.config.decoder_config
    for k, v in (
        ("hidden_size", target_hidden),
        ("intermediate_size", llm.config.intermediate_size),
        ("num_attention_heads", llm.config.num_attention_heads),
        ("num_key_value_heads", llm.config.num_key_value_heads),
        ("num_hidden_layers", llm.config.num_hidden_layers),
        ("head_dim", getattr(llm.config, "head_dim", 128)),
        ("tie_word_embeddings", True),
    ):
        setattr(dc, k, v)

    del llm
    return student, connector_method, target_hidden


def peak_gib(device: str) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=str(ROOT / "models/teacher"))
    ap.add_argument("--student-llm", default=str(ROOT / "models/student_1p5b_tied_smoke"))
    ap.add_argument("--tokenizer", default=str(ROOT / "models/tokenizer"))
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/simulated.jsonl"))
    ap.add_argument("--student-device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )

    # The processor only accepts a tokenizer path containing 'qwen' (see
    # VibeVoiceASRProcessor.from_pretrained). Our local Qwen2.5 tokenizer lives
    # at models/tokenizer; expose it under a qwen-named symlink if needed.
    tok_path = args.tokenizer
    if "qwen" not in Path(tok_path).name.lower():
        import os
        import tempfile

        link = Path(tempfile.gettempdir()) / "qwen_tokenizer_09b"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
        except OSError:
            pass
        os.symlink(os.path.abspath(tok_path), link)
        tok_path = str(link)
    from distil_vibevoice.data.manifest import read_manifest, format_target
    from distil_vibevoice.distill.collator import DistillCollator
    from distil_vibevoice.distill.losses import distill_loss

    dtype = torch.bfloat16
    dev_s, dev_t = args.student_device, args.teacher_device

    report = {
        "student_assembled": False,
        "connector_method": "",
        "forward_ok": False,
        "logit_shapes_aligned": False,
        "distill_ran": False,
        "steps": 0,
        "loss_first": float("nan"),
        "loss_last": float("nan"),
        "peak_gib_cuda0": 0.0,
        "peak_gib_cuda1": 0.0,
        "blocker": "",
        "notes": "",
    }

    torch.cuda.reset_peak_memory_stats(dev_s)
    torch.cuda.reset_peak_memory_stats(dev_t)

    # --- processor + one real clip -----------------------------------------
    processor = VibeVoiceASRProcessor.from_pretrained(
        args.teacher, language_model_pretrained_name=tok_path
    )
    records = read_manifest(args.manifest)
    rec = records[0]
    wav = load_audio(rec.audio_path, max_sec=args.max_sec)
    dur = len(wav) / 24000.0
    hotwords = "\n".join(
        dict.fromkeys(s.text for s in rec.segments if s.text)
    )[:512]

    prompt = processor(
        audio=[wav],
        context_info=hotwords,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    prompt_ids = prompt["input_ids"][0].tolist()
    prompt_acmask = prompt["acoustic_input_mask"][0].tolist()
    speech_tensors = prompt["speech_tensors"]
    speech_masks = prompt["speech_masks"]

    target_text = format_target(rec.segments)
    tok = processor.tokenizer
    tgt_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    tail = tgt_ids + ([eos] if eos is not None else [])
    full_ids = prompt_ids + tail
    full_labels = [-100] * len(prompt_ids) + tail

    collator = DistillCollator(tok, max_len=16384)
    batch = collator([{"input_ids": full_ids, "labels": full_labels}])
    S = batch["input_ids"].shape[1]
    acmask = torch.zeros(1, S, dtype=torch.bool)
    acmask[0, : len(prompt_acmask)] = torch.tensor(prompt_acmask, dtype=torch.bool)
    n_audio = int(acmask.sum().item())
    report["notes"] += (
        f"clip={Path(rec.audio_path).name} dur={dur:.1f}s S={S} "
        f"audio_frames={n_audio} tgt_tok={len(tgt_ids)}; "
    )

    # --- load teacher (cuda:1) ---------------------------------------------
    teacher = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.teacher, dtype=dtype, attn_implementation="sdpa"
    ).to(dev_t).eval()

    # --- assemble student (CPU -> cuda:0) ----------------------------------
    try:
        student, connector_method, hid = build_student(
            args.teacher, args.student_llm, dtype
        )
        student = student.to(dev_s)
        report["connector_method"] = connector_method
        report["student_assembled"] = True
        report["notes"] += f"student_hidden={hid}; "
    except Exception as e:
        report["blocker"] = f"assembly failed: {type(e).__name__}: {e}"
        _finish(report, args)
        return 0

    def run(model, device):
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            speech_tensors=speech_tensors.to(device),
            speech_masks=speech_masks.to(device),
            acoustic_input_mask=acmask.to(device),
            use_cache=False,
        ).logits

    # --- single forward gate -----------------------------------------------
    try:
        with torch.no_grad():
            t_logits = run(teacher, dev_t)
        with torch.no_grad():
            s_logits = run(student, dev_s)
        aligned = (
            s_logits.shape[1] == t_logits.shape[1]
            and s_logits.shape[-1] == 152064
            and s_logits.shape[-1] == t_logits.shape[-1]
        )
        finite = bool(torch.isfinite(s_logits).all() and torch.isfinite(t_logits).all())
        report["forward_ok"] = finite
        report["logit_shapes_aligned"] = bool(aligned)
        report["notes"] += (
            f"student_logits={tuple(s_logits.shape)} "
            f"teacher_logits={tuple(t_logits.shape)} finite={finite}; "
        )
        if not (aligned and finite):
            report["blocker"] = "forward produced misaligned or non-finite logits"
            _finish(report, args)
            return 0
    except Exception as e:
        report["blocker"] = f"forward failed: {type(e).__name__}: {e}"
        _finish(report, args)
        return 0

    # --- distill smoke ------------------------------------------------------
    try:
        teacher_soft = t_logits.detach().to(dev_s)  # bf16 full-vocab soft targets
        labels = batch["labels"].to(dev_s)
        tw = batch["token_weights"].to(dev_s)
        # causal shift: logits[:, :-1] predicts labels[:, 1:]
        t_soft = teacher_soft[:, :-1, :]
        lab = labels[:, 1:]
        tw_s = tw[:, 1:]

        # Freeze encoders; train LLM + connectors + lm_head (connectors get no
        # grad due to no_grad encode path, so effectively LLM + head + embeds).
        for p in student.model.acoustic_tokenizer.parameters():
            p.requires_grad_(False)
        for p in student.model.semantic_tokenizer.parameters():
            p.requires_grad_(False)
        student.train()
        student.model.acoustic_tokenizer.eval()
        student.model.semantic_tokenizer.eval()

        seen, params = set(), []
        for p in student.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        opt = torch.optim.AdamW(params, lr=args.lr)

        losses = []
        t0 = time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            s_log = run(student, dev_s)
            out = distill_loss(
                s_log[:, :-1, :],
                t_soft,
                lab,
                token_weights=tw_s,
                kept_vocab_ids=None,
            )
            loss = out["loss"]
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        dt = time.time() - t0

        report["distill_ran"] = True
        report["steps"] = args.steps
        report["loss_first"] = losses[0]
        report["loss_last"] = losses[-1]
        report["notes"] += (
            f"sec/step={dt/max(args.steps,1):.2f} "
            f"kl0={float(out['kl']):.3f} ce0={float(out['ce']):.3f}; "
        )
    except Exception as e:
        report["blocker"] = f"distill failed: {type(e).__name__}: {e}"

    _finish(report, args)
    return 0


def _finish(report: dict, args) -> None:
    report["peak_gib_cuda0"] = round(peak_gib("cuda:0"), 2)
    report["peak_gib_cuda1"] = round(peak_gib("cuda:1"), 2)
    report["loss_first"] = round(report["loss_first"], 4) if report["loss_first"] == report["loss_first"] else report["loss_first"]
    report["loss_last"] = round(report["loss_last"], 4) if report["loss_last"] == report["loss_last"] else report["loss_last"]
    import json

    print("RESULT_JSON " + json.dumps(report, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
