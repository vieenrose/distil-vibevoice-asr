#!/usr/bin/env python
"""FT v5: fix the diarization erosion measured across v2->v3->v4.

DIAGNOSIS (this session, held-out IVOD 5/30/123-min vs base MOSS):
  single-pass speaker tags on the 5-min meeting (ref = 3 speakers):
    base=3  ->  v2=2  ->  v3=1  ->  v4=1
  i.e. each transcription-focused FT round progressively FORGOT the base
  model's speaker separation. The training data is NOT the cause (TTS ~3.5
  spk/rec, IVOD ~6). Timestamps were unaffected. Root cause: [Sxx] speaker
  tags are a tiny fraction of the target tokens, so a plain uniform CE can
  drive transcription loss down while letting the speaker tags decay.

FIX (this script): a per-token WEIGHTED cross-entropy that up-weights the
`[Sxx]` speaker-tag tokens (default 8x). The model can no longer cheaply drop
speaker identity — getting the tag wrong now costs 8x the loss. Everything
else (300s windows, RIR+MUSAN aug, IVOD mix, LR/cosine) matches v4 (scripts
/39) so this is a clean ablation: the ONLY change is defending diarization.

Start from v2 by default (best diarization of the FT checkpoints + strong
zh-TW), continue at 300s windows so long-window behavior stays in-distribution.
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


def target_text(segments) -> str:
    return "".join(f"[{s['start']:.2f}][S{s['speaker']:02d}]{s['text']}"
                   f"[{s['end']:.2f}]" for s in segments)


def speaker_token_weights(tok, assistant: str, label_ids: list[int],
                          eos_id: int, w_spk: float):
    """Per-token weight aligned to the non-masked (assistant + eos) labels.

    Returns None if the standalone assistant tokenization doesn't match the
    label ids (BPE boundary drift) -> caller falls back to uniform weighting
    for that step, so a mismatch never corrupts the loss.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v2")
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.3)
    ap.add_argument("--p-aug", type=float, default=0.43)
    ap.add_argument("--rir-dir", default="data/aug/RIRS_NOISES")
    ap.add_argument("--musan-dir", default="data/aug/musan")
    ap.add_argument("--spk-weight", type=float, default=8.0,
                    help="CE weight on [Sxx] speaker-tag tokens (1.0 = v4)")
    ap.add_argument("--kl-base", default="",
                    help="path to a frozen teacher (e.g. models/moss) to KL-anchor "
                         "the speaker-tag token DISTRIBUTION to. Transfers voice->"
                         "speaker discrimination that hard-target CE can't (the "
                         "model games CE via turn-taking cues + stays voice-collapsed).")
    ap.add_argument("--kl-weight", type=float, default=2.0,
                    help="weight on the speaker-position KL(student||teacher)")
    ap.add_argument("--fake-quant", action="store_true",
                    help="q4 QAT: wrap decoder Linears with int4 STE fake-quant. "
                         "Keeps the speaker-weighted loss so quantization-adapt "
                         "does NOT re-collapse diarization (use LR ~1e-5).")
    ap.add_argument("--exclude-sensitivity", default="data/q4_sensitivity.json")
    ap.add_argument("--keep-int8-top", type=int, default=0,
                    help="keep the N most q4-sensitive layers at int8 (mixed-prec)")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-audio-s", type=float, default=300.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default="models/moss_ft_zhtw_v5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    from distil_vibevoice.data.augment import augment_wav

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos_id = tok.eos_token_id

    teacher = None
    if args.kl_base:
        teacher = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.kl_base) if not Path(args.kl_base).is_absolute() else args.kl_base,
            trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"KL-anchor: frozen teacher {args.kl_base} on speaker positions "
              f"(kl-weight={args.kl_weight})", flush=True)

    if args.fake_quant:
        import re as _re
        from distil_vibevoice.quant.fakequant import wrap_decoder_linears
        exclude_frag = set()
        sp = ROOT / args.exclude_sensitivity
        if args.keep_int8_top > 0 and sp.exists():
            ranked = json.loads(sp.read_text())["ranked"]
            exclude_frag = {k for k, _ in ranked[:args.keep_int8_top]}
            print(f"keeping int8 (excluded from QAT): {sorted(exclude_frag)}", flush=True)

        def _excluded(name: str) -> bool:
            if "lm_head" in name and "lm_head" in exclude_frag:
                return True
            m = _re.search(r"layers\.(\d+)\.", name)
            return bool(m) and f"layer{int(m.group(1)):02d}" in exclude_frag

        excl = {n for n, _ in model.named_modules() if _excluded(n)}
        wrapped = wrap_decoder_linears(model, exclude=excl)
        print(f"QAT: wrapped {len(wrapped)} decoder Linears with int4 fake-quant "
              f"(spk-weight={args.spk_weight} still ON -> diarization-defended QAT)",
              flush=True)

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
    print(f"TTS {len(tts_rows)} | IVOD {len(ivod_rows)} meetings; "
          f"{args.steps} steps @ {args.max_audio_s:.0f}s; "
          f"spk-weight={args.spk_weight}", flush=True)

    tgt_sr = proc.feature_extractor.sampling_rate
    losses, n_skip, n_uniform = [], 0, 0
    spk_frac_log = []
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
            aug = True
        else:
            aug = False

        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

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

        # per-token weight vector aligned to the full sequence
        label_ids = input_ids[0, n_prompt:].tolist()
        w_assist = speaker_token_weights(tok, assistant, label_ids, eos_id,
                                         args.spk_weight)
        weight_vec = torch.ones_like(input_ids, dtype=torch.float32)
        if w_assist is not None and len(w_assist) == len(label_ids):
            weight_vec[0, n_prompt:] = torch.tensor(w_assist, device=dev)
            spk_frac_log.append(sum(1 for x in w_assist if x > 1.0)
                                / max(1, len(w_assist)))
        else:
            n_uniform += 1  # fell back to uniform for this step

        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in enc.items()}

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        out = model(**batch)
        logits = out.logits  # [1, L, V] bf16
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_w = weight_vec[:, 1:]
        # keep bf16 (cross_entropy upcasts internally per-row) — an explicit
        # .float() on the full [L, 152k] logits materialized a ~3GB fp32 copy
        # that OOM'd the 2-model (student+teacher) setup on long windows.
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(shift_labels.shape).float()
        mask = (shift_labels != -100).float()
        denom = (shift_w * mask).sum().clamp_min(1.0)
        loss = (ce * shift_w * mask).sum() / denom

        # KL-anchor the speaker-token distribution to the frozen teacher (base),
        # ONLY at speaker-tag positions (weight_vec>1) — transfers base's intact
        # voice->speaker discrimination without touching content tokens (so the
        # student's zh-TW transcription is unaffected). Soft distribution carries
        # the "different voice = different speaker" signal that hard-target CE
        # lacks, which is why plain weighted CE couldn't restore diarization.
        kl_val = 0.0
        if teacher is not None:
            spk_pos = (shift_w > 1.0) & (shift_labels != -100)
            if spk_pos.any():
                with torch.no_grad():
                    t_logits = teacher(**batch).logits[:, :-1, :]
                s_lp = F.log_softmax(shift_logits[spk_pos].float(), dim=-1)
                t_p = F.softmax(t_logits[spk_pos].float(), dim=-1)
                kl = F.kl_div(s_lp, t_p, reduction="batchmean")
                loss = loss + args.kl_weight * kl
                kl_val = float(kl.detach())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 10 == 0:
            src = "ivod" if use_ivod else ("tts+aug" if aug else "tts")
            sf_ = (sum(spk_frac_log[-50:]) / len(spk_frac_log[-50:])
                   if spk_frac_log else 0.0)
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e} "
                  f"[{src}] len={input_ids.shape[1]} spk_tok_frac={sf_:.3f}"
                  f"{f' kl={kl_val:.3f}' if teacher is not None else ''}",
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
          f"(skipped {n_skip}, uniform-fallback {n_uniform}) | saved -> {outdir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
