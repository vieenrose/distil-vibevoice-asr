#!/usr/bin/env python
"""Streaming FT: teach MOSS to decode with a BOUNDED audio-KV window.

MOTIVATION (measured, this project): monotonic audio-KV eviction at inference
(RS_AUDIO_KV_WINDOW=45 in RapidSpeech.cpp) makes the model drift to Simplified
Chinese and stop early — MOSS was trained with FULL attention over all audio
tokens, so chopping the audio KV is out-of-distribution. This script closes
that gap: fine-tune with a 4D attention mask that EMULATES the eviction
(text tokens of a segment starting at time T cannot attend audio tokens older
than T - W seconds), while a frozen full-attention teacher (the same
checkpoint) KL-distills its distribution into the windowed student.

Keeps the v5 protections: speaker-tag-weighted CE (8x) so diarization does not
collapse, and the teacher KL doubles as the anchor-to-base defence.

Deployment target: RapidSpeech.cpp's env-gated eviction path (positions are
PRESERVED under eviction there — logical_pos keeps counting — which matches
training with a mask and ordinary consecutive position_ids).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

SPK_RE = re.compile(r"\[S\d+\]")
AUDIO_TOK_DT = 0.08          # decoder-side audio tokens: 12.5 / s


def window_sample(rec: dict, max_s: float, rng: random.Random):
    dur = float(rec.get("duration") or max(s["end"] for s in rec["segments"]))
    inside = []
    for _ in range(8):
        off = 0.0 if dur <= max_s else rng.uniform(0.0, dur - max_s)
        inside = [s for s in rec["segments"]
                  if s["start"] >= off and s["end"] <= off + max_s]
        if len(inside) >= 2 or dur <= max_s:
            break
    if not inside:
        return None, None
    spk_map: dict = {}
    segs = []
    for s in sorted(inside, key=lambda x: x["start"]):
        if s["speaker"] not in spk_map:
            spk_map[s["speaker"]] = len(spk_map) + 1
        segs.append({"start": s["start"] - off, "end": s["end"] - off,
                     "speaker": spk_map[s["speaker"]], "text": s["text"]})
    return off, segs


def seg_text(s) -> str:
    return f"[{s['start']:.2f}][S{s['speaker']:02d}]{s['text']}[{s['end']:.2f}]"


def target_text(segments) -> str:
    return "".join(seg_text(s) for s in segments)


def speaker_token_weights(tok, assistant: str, label_ids: list[int],
                          eos_id: int, w_spk: float):
    """Per-token weight aligned to the non-masked (assistant + eos) labels.
    None on BPE drift -> caller falls back to uniform for that step."""
    enc = tok(assistant, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if ids + [eos_id] != list(label_ids):
        return None
    spans = [(m.start(), m.end()) for m in SPK_RE.finditer(assistant)]
    w = []
    for a, b in offs:
        hit = a != b and any(a < e and b > s for s, e in spans)
        w.append(w_spk if hit else 1.0)
    w.append(1.0)  # eos
    return w


def eviction_bias(input_ids, audio_token_id, n_prompt, tok, segs,
                  window_s: float, dtype, device):
    """(1,1,L,L) additive attention bias emulating monotonic audio-KV eviction.

    Causal everywhere; ADDITIONALLY, target-text tokens belonging to a segment
    that starts at time T cannot see audio tokens with timestamp < T - W (they
    were evicted by the time the decoder got there). Prompt/audio/suffix
    positions keep plain causal attention (prefill happens before eviction).

    Returns None if the per-segment tokenization doesn't concatenate to the
    exact label ids (BPE drift) -> caller trains that step with full attention
    rather than a misaligned mask.
    """
    L = input_ids.shape[1]
    ids0 = input_ids[0]
    audio_pos = (ids0 == audio_token_id).nonzero().flatten()
    if audio_pos.numel() == 0:
        return None
    a0 = audio_pos[0].item()
    audio_t = (audio_pos - a0).to(torch.float32) * AUDIO_TOK_DT

    # Map target tokens -> segments via offset_mapping on the WHOLE assistant
    # string (per-seg tokenization does NOT concatenate: BPE merges across the
    # "[end][start]" boundary). A token straddling a boundary goes to the
    # EARLIER segment (smaller cutoff = less eviction = conservative).
    assistant = "".join(seg_text(s) for s in segs)
    enc = tok(assistant, return_offsets_mapping=True, add_special_tokens=False)
    if n_prompt + len(enc["input_ids"]) + 1 != L:          # + eos
        return None
    if enc["input_ids"] != ids0[n_prompt:L - 1].tolist():  # BPE drift guard
        return None
    starts, cum = [], 0
    for s in segs:
        starts.append((cum, float(s["start"])))
        cum += len(seg_text(s))
    spans = []           # (token_start, token_end, seg_start_time), contiguous
    cur_seg, run_start = 0, n_prompt
    for i, (a, _b) in enumerate(enc["offset_mapping"]):
        seg = cur_seg
        while seg + 1 < len(starts) and a >= starts[seg + 1][0]:
            seg += 1
        if seg != cur_seg:
            spans.append((run_start, n_prompt + i, starts[cur_seg][1]))
            cur_seg, run_start = seg, n_prompt + i
    spans.append((run_start, L, starts[cur_seg][1]))       # tail + eos

    allow = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
    for qs, qe, t_start in spans:
        cutoff = t_start - window_s
        if cutoff <= 0:
            continue
        dead = audio_pos[audio_t < cutoff]
        if dead.numel():
            allow[qs:qe, dead] = False
    bias = torch.zeros(L, L, dtype=dtype, device=device)
    bias.masked_fill_(~allow, torch.finfo(dtype).min)
    return bias.unsqueeze(0).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v5_kl_qat_fixed")
    ap.add_argument("--kl-base", default="models/moss_ft_zhtw_v5_kl_qat_fixed",
                    help="frozen FULL-ATTENTION teacher for the windowed student")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="KL(student||teacher) on ALL label positions")
    ap.add_argument("--spk-kl-weight", type=float, default=2.0,
                    help="EXTRA KL on [Sxx] speaker-tag positions (v5 lesson: "
                         "all-position KL alone lets diarization drift)")
    ap.add_argument("--window-s", type=float, default=45.0,
                    help="audio-KV window the student must live with")
    ap.add_argument("--window-anneal", default="120:300,75:800",
                    help="'W:until,...' curriculum before settling on --window-s")
    ap.add_argument("--spk-weight", type=float, default=8.0)
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.5,
                    help="long real meetings exercise the window most")
    ap.add_argument("--p-aug", type=float, default=0.43)
    ap.add_argument("--p-silence-tail", type=float, default=0.0,
                    help="prob of trimming the wav at the last segment end and "
                         "appending 15-60s of near-silence (target unchanged: "
                         "the model must NOT invent speech after the talk ends)")
    ap.add_argument("--rir-dir", default="data/aug/RIRS_NOISES")
    ap.add_argument("--musan-dir", default="data/aug/musan")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-audio-s", type=float, default=300.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default="models/moss_ft_zhtw_v6_stream")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    from distil_vibevoice.data.augment import augment_wav

    anneal = []
    if args.window_anneal:
        for part in args.window_anneal.split(","):
            w, until = part.split(":")
            anneal.append((int(until), float(w)))
        anneal.sort()

    def window_at(step):
        for until, w in anneal:
            if step < until:
                return w
        return args.window_s

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos_id = tok.eos_token_id
    audio_token_id = model.config.audio_token_id

    teacher = AutoModelForCausalLM.from_pretrained(
        args.kl_base, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher (full attention, frozen): {args.kl_base} "
          f"kl-weight={args.kl_weight}", flush=True)

    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    tts_rows = []
    for mp in args.tts_manifests:
        p = ROOT / mp
        if p.exists():
            tts_rows += [json.loads(l) for l in p.open()]
    ivod_rows = []
    p = ROOT / args.ivod_manifest
    if p.exists():
        ivod_rows = [json.loads(l) for l in p.open()]
    if not ivod_rows:
        print("WARNING: no IVOD rows; TTS-only (+aug)")
        args.p_ivod = 0.0
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rng.shuffle(tts_rows)
    rng.shuffle(ivod_rows)
    print(f"TTS {len(tts_rows)} | IVOD {len(ivod_rows)}; {args.steps} steps @ "
          f"{args.max_audio_s:.0f}s; W={args.window_s}s anneal={args.window_anneal}",
          flush=True)

    tgt_sr = proc.feature_extractor.sampling_rate
    losses, n_skip, n_uniform, n_fullattn = [], 0, 0, 0
    ti = ii = 0
    for step in range(args.steps):
        use_ivod = ivod_rows and rng.random() < args.p_ivod
        if use_ivod:
            rec = ivod_rows[ii % len(ivod_rows)]; ii += 1
        else:
            rec = tts_rows[ti % len(tts_rows)]; ti += 1

        off, segs = window_sample(rec, args.max_audio_s, rng)
        if not segs:
            n_skip += 1
            continue
        try:
            wav, sr = sf.read(rec["audio_path"], start=int(off * 24000),
                              frames=int(args.max_audio_s * 24000))
        except Exception:
            wav, sr = sf.read(rec["audio_path"])
            wav = wav[int(off * sr): int((off + args.max_audio_s) * sr)]
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1),
                         dtype=np.float32)
        if len(wav) < sr:
            n_skip += 1
            continue
        if (not use_ivod) and rng.random() < args.p_aug:
            wav = augment_wav(wav, sr, rir_dir=args.rir_dir,
                              musan_dir=args.musan_dir, rng=np_rng)
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        if rng.random() < args.p_silence_tail:
            cut = int(min(len(wav), (segs[-1]["end"] + rng.uniform(0.3, 2.0))
                          * tgt_sr))
            tail = int(rng.uniform(15.0, 60.0) * tgt_sr)
            noise = (np_rng.standard_normal(tail) * 1e-4).astype(np.float32)
            wav = np.concatenate([wav[:cut], noise])

        assistant = target_text(segs)
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        prompt_text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + assistant + tok.eos_token

        enc = proc(text=full_text, audio=[wav], max_length=args.max_len,
                   return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        n_prompt = proc(text=prompt_text, audio=[wav], max_length=args.max_len,
                        return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100

        label_ids = input_ids[0, n_prompt:].tolist()
        w_assist = speaker_token_weights(tok, assistant, label_ids, eos_id,
                                         args.spk_weight)
        weight_vec = torch.ones_like(input_ids, dtype=torch.float32)
        if w_assist is not None and len(w_assist) == len(label_ids):
            weight_vec[0, n_prompt:] = torch.tensor(w_assist, device=dev)
        else:
            n_uniform += 1

        W = window_at(step)
        bias = eviction_bias(input_ids, audio_token_id, n_prompt, tok, segs,
                             W, torch.bfloat16, dev)
        if bias is None:
            n_fullattn += 1          # BPE drift: train full-attention this step

        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in enc.items()}
        if bias is not None:
            batch["attention_mask"] = bias

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        out = model(**batch)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_w = weight_vec[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(shift_labels.shape).float()
        mask = (shift_labels != -100).float()
        denom = (shift_w * mask).sum().clamp_min(1.0)
        loss = (ce * shift_w * mask).sum() / denom

        # Full-attention teacher -> windowed student, ALL label positions:
        # carries both the transcription distribution the student can no longer
        # read off evicted audio, and the v5 speaker-discrimination anchor.
        kl_val = 0.0
        if args.kl_weight > 0:
            lbl_pos = shift_labels != -100
            with torch.no_grad():
                t_batch = {k: v for k, v in batch.items()
                           if k != "attention_mask"}
                t_logits = teacher(**t_batch).logits[:, :-1, :]
            s_lp = F.log_softmax(shift_logits[lbl_pos].float(), dim=-1)
            t_p = F.softmax(t_logits[lbl_pos].float(), dim=-1)
            kl = F.kl_div(s_lp, t_p, reduction="batchmean")
            loss = loss + args.kl_weight * kl
            kl_val = float(kl.detach())
            # v5 lesson: without a DEDICATED speaker-position KL the [Sxx]
            # distribution drifts (DER 0.128 -> 0.190 measured on the first
            # streaming FT) even with 8x CE weight + all-position KL.
            spk_pos = (shift_w > 1.0) & lbl_pos
            if args.spk_kl_weight > 0 and spk_pos.any():
                s_spk = F.log_softmax(shift_logits[spk_pos].float(), dim=-1)
                t_spk = F.softmax(t_logits[spk_pos].float(), dim=-1)
                loss = loss + args.spk_kl_weight * F.kl_div(
                    s_spk, t_spk, reduction="batchmean")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 10 == 0:
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e} "
                  f"W={W:.0f}s len={input_ids.shape[1]} kl={kl_val:.3f}",
                  flush=True)
        if args.save_every and step and step % args.save_every == 0:
            ck = ROOT / (args.out + f"_ckpt{step}")
            model.save_pretrained(ck)
            proc.save_pretrained(ck)
            print(f"  ckpt -> {ck}", flush=True)

    outdir = ROOT / args.out
    model.save_pretrained(outdir)
    proc.save_pretrained(outdir)
    print(f"\nloss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f} "
          f"(skip {n_skip}, uniform {n_uniform}, fullattn-fallback {n_fullattn})"
          f" | saved -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
