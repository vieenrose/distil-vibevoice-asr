#!/usr/bin/env python
"""Ternary-quantize the Whisper encoder losslessly — recipe from arXiv
2505.21245 ("Towards One-bit ASR", CUHK 2025) mapped onto MOSS-TD.

Why the encoder: the paper achieves statistically LOSSLESS 1-2 bit Conformer
*encoders* at ordinary training budgets, while AR decoders need ~10B-token
continued pretraining (BitDistill) — our decoder-ternary attempts hit exactly
that wall (CER 11%→21%). Encoder Linears are 302M/307M params.

The three ingredients we were missing in scripts/52:
  1. LEARNABLE tensor-wise scale α (their Eq. 3 gradient — realized exactly by
     autograd through STE-round + clamp; fixed absmean scales are the single
     biggest accuracy leak below 3 bits per ParetoQ).
  2. Weight-sharing CO-TRAINING with a PROXIMAL teacher: the ternary model is
     the q4 model re-quantized — same latent weights, two quant functions per
     step, L = L_q4 + 0.5·L_tern + 1.0·KL(SG(p_q4)||p_tern). We previously
     distilled from fp (gap too large) or trained ternary alone.
  3. STOCHASTIC PRECISION: on alternating steps the ternary branch quantizes
     only a random per-layer subset (prob log-linear 0.2→0.9 by depth),
     a smooth continuum instead of our cliff-prone hard anneal.

Conv stem / positional / VQ-adaptor / decoder stay untouched (paper: CNN
quantization hurts most; decoder ships q4 as-is).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- quant ------
def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


class CoQuantLinear(nn.Module):
    """Weight-sharing dual-precision Linear.

    mode 'q4'  : block-32 int4 fake-quant of the latent weight (anchor branch,
                 matches scripts/52 semantics).
    mode 'tern': ternary with LEARNABLE tensor-wise scale α:
                 Ŵ = α · round(clamp(W/α, -1, 1)); autograd through the STE
                 yields exactly Eq. 3 of 2505.21245 (−W/α+Π inside the clip,
                 sign(W/α) at saturation).
    mode 'fp'  : bypass (stochastic-precision steps leave some layers fp… no —
                 SP leaves non-sampled layers at the TEACHER precision, q4).
    """

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin
        # α init: E|W| (absmean — the BitNet-style scale, now trainable).
        self.alpha = nn.Parameter(lin.weight.detach().abs().mean().float().clone())
        self.mode = "q4"

    def _w_q4(self) -> torch.Tensor:
        from distil_vibevoice.quant.fakequant import fake_quant_kbit
        return fake_quant_kbit(self.lin.weight, 4, 32)

    def _w_tern(self) -> torch.Tensor:
        a = self.alpha.abs().clamp_min(1e-8).to(self.lin.weight.dtype)
        return a * ste_round(torch.clamp(self.lin.weight / a, -1.0, 1.0))

    def forward(self, x):
        w = self._w_tern() if self.mode == "tern" else self._w_q4()
        return F.linear(x, w.to(x.dtype), self.lin.bias)


ENC_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def wrap_encoder(model):
    wrapped = []
    for name, module in list(model.named_modules()):
        if "whisper_encoder.layers" not in name:
            continue
        for cn, child in list(module.named_children()):
            if cn in ENC_TARGETS and isinstance(child, nn.Linear):
                cq = CoQuantLinear(child)
                setattr(module, cn, cq)
                m = re.search(r"layers\.(\d+)\.", name + ".")
                wrapped.append((int(m.group(1)) if m else -1, cq))
    return wrapped


# --------------------------------------------------------------- main -------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v6_stream3")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-tern", type=float, default=0.5)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--init-patch", default="",
                    help="warm-start latents+alphas from a prior enc_ternary_patch.pt")
    ap.add_argument("--three-branch", action="store_true",
                    help="paper Eq.7: anchor + full-ternary + SP branches every step")
    ap.add_argument("--sp-every", type=int, default=2,
                    help="every Nth step the ternary branch samples a random "
                         "per-layer subset (prob 0.2→0.9 log-linear by depth)")
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.4)
    ap.add_argument("--max-audio-s", type=float, default=120.0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="models/moss_enc_ternary")
    ap.add_argument("--save-every", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer

    wrapped = wrap_encoder(model)
    n_layer_enc = 1 + max(l for l, _ in wrapped)
    print(f"co-quant wrapped {len(wrapped)} encoder Linears "
          f"({n_layer_enc} layers)", flush=True)
    if args.init_patch:
        pk = torch.load(args.init_patch, map_location="cpu", weights_only=False)
        for i, (_, cq) in enumerate(wrapped):
            cq.lin.weight.data.copy_(pk["tensors"][f"{i:03d}"].to(dev))
            cq.alpha.data.copy_(pk["alphas"][f"{i:03d}"].to(dev).float())
        print(f"warm-started from {args.init_patch}", flush=True)

    # Freeze everything except encoder latents + alphas.
    for p in model.parameters():
        p.requires_grad_(False)
    train_params, alpha_params = [], []
    for _, cq in wrapped:
        cq.lin.weight.requires_grad_(True)
        cq.alpha.requires_grad_(True)  # freeze-all above caught the alphas too
        train_params.append(cq.lin.weight)
        alpha_params.append(cq.alpha)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW(
        [{"params": train_params, "lr": args.lr},
         {"params": alpha_params, "lr": args.lr}])

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # Depth-ramped stochastic-precision probs (log-linear 0.2 → 0.9).
    sp_prob = [0.2 * (0.9 / 0.2) ** (l / max(1, n_layer_enc - 1))
               for l in range(n_layer_enc)]

    def set_modes(mode_by_layer):
        for l, cq in wrapped:
            cq.mode = mode_by_layer(l)

    # ---- data (loader semantics of scripts/52) ----
    def window_sample(rec, max_s, rng):
        dur = float(rec.get("duration") or max(s["end"] for s in rec["segments"]))
        off = 0.0 if dur <= max_s else rng.uniform(0, dur - max_s)
        inside = [s for s in rec["segments"]
                  if s["start"] >= off and s["end"] <= off + max_s]
        if not inside:
            return None, None
        spk_map, segs = {}, []
        for s in sorted(inside, key=lambda x: x["start"]):
            if s["speaker"] not in spk_map:
                spk_map[s["speaker"]] = len(spk_map) + 1
            segs.append({"start": s["start"] - off, "end": s["end"] - off,
                         "speaker": spk_map[s["speaker"]], "text": s["text"]})
        return off, segs

    def target_text(segs):
        return "".join(f"[{s['start']:.2f}][S{s['speaker']:02d}]{s['text']}"
                       f"[{s['end']:.2f}]" for s in segs)

    tts = []
    for mp in args.tts_manifests:
        p = ROOT / mp
        if p.exists():
            tts += [json.loads(l) for l in p.open()]
    ivod = ([json.loads(l) for l in (ROOT / args.ivod_manifest).open()]
            if (ROOT / args.ivod_manifest).exists() else [])
    rng = random.Random(args.seed)
    rng.shuffle(tts); rng.shuffle(ivod)
    tgt_sr = proc.feature_extractor.sampling_rate
    print(f"TTS {len(tts)} | IVOD {len(ivod)}; {args.steps} steps @ "
          f"{args.max_audio_s:.0f}s", flush=True)

    losses, ti, ii = [], 0, 0
    for step in range(args.steps):
        use_ivod = ivod and rng.random() < args.p_ivod
        rec = (ivod[ii % len(ivod)] if use_ivod else tts[ti % len(tts)])
        if use_ivod: ii += 1
        else: ti += 1
        off, segs = window_sample(rec, args.max_audio_s, rng)
        if not segs:
            continue
        try:
            wav, sr = sf.read(rec["audio_path"], start=int(off * 24000),
                              frames=int(args.max_audio_s * 24000))
        except Exception:
            wav, sr = sf.read(rec["audio_path"])
            wav = wav[int(off * sr): int((off + args.max_audio_s) * sr)]
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1), np.float32)
        if len(wav) < sr:
            continue
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        ptext = proc.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        enc = proc(text=ptext + target_text(segs) + tok.eos_token,
                   audio=[wav], max_length=args.max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        npf = proc(text=ptext, audio=[wav], max_length=args.max_len,
                   return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :npf] = -100
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in enc.items()}
        batch["labels"] = labels

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # Per-branch backward BEFORE switching modes: gradient checkpointing
        # replays the forward during backward, so module modes must still
        # match that branch's forward when its backward runs. Accumulation
        # across the two backwards is equivalent to one combined loss.
        opt.zero_grad()

        # ---- branch A: q4 anchor (teacher) ----
        set_modes(lambda l: "q4")
        out_a = model(**batch)
        loss_a = out_a.loss
        p_teach = out_a.logits.detach()
        loss_a.backward()
        loss_a = loss_a.detach()

        lbl = labels[:, 1:] != -100
        t_p = F.softmax(p_teach[:, :-1][lbl].float(), -1)

        def student_pass(mode_fn):
            set_modes(mode_fn)
            out = model(**batch)
            s_lp = F.log_softmax(out.logits[:, :-1][lbl].float(), -1)
            k = F.kl_div(s_lp, t_p, reduction="batchmean")
            (args.lambda_tern * out.loss + args.lambda_kl * k).backward()
            return out.loss.detach(), k.detach()

        if args.three_branch:
            # paper Eq. 7: full-ternary branch AND SP branch every step
            loss_b, kl = student_pass(lambda l: "tern")
            mask = [rng.random() < sp_prob[l] for l in range(n_layer_enc)]
            loss_c, kl_c = student_pass(lambda l: "tern" if mask[l] else "q4")
        elif args.sp_every > 0 and step % args.sp_every == 1:
            mask = [rng.random() < sp_prob[l] for l in range(n_layer_enc)]
            loss_b, kl = student_pass(lambda l: "tern" if mask[l] else "q4")
        else:
            loss_b, kl = student_pass(lambda l: "tern")
        torch.nn.utils.clip_grad_norm_(train_params + alpha_params, 1.0)
        opt.step()
        losses.append(loss_b.detach().item())
        if step % 20 == 0:
            am = torch.stack([a.detach().float() for a in alpha_params])
            print(f"  step {step}: q4={loss_a.item():.3f} "
                  f"tern={loss_b.item():.3f} kl={kl.item():.3f} "
                  f"lr={lr_at(step):.2e} alpha_mu={am.mean():.4f}", flush=True)
        if args.save_every and step and step % args.save_every == 0:
            _save(args, wrapped, step)

    _save(args, wrapped, None)
    print(f"\ntern loss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f}",
          flush=True)
    return 0


def _save(args, wrapped, step):
    outdir = ROOT / (args.out + (f"_ckpt{step}" if step else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    patch = {}
    for l, cq in wrapped:
        # keys resolved at eval time by module identity order
        patch[f"layer{l:02d}.{id(cq)}"] = None
    tensors, alphas, names = {}, {}, []
    # re-walk to persist with real names
    return_path = outdir / "enc_ternary_patch.pt"
    torch.save({"base": args.model,
                "tensors": {f"{i:03d}": cq.lin.weight.detach().to(torch.bfloat16).cpu()
                            for i, (_, cq) in enumerate(wrapped)},
                "alphas": {f"{i:03d}": cq.alpha.detach().float().cpu()
                           for i, (_, cq) in enumerate(wrapped)},
                "layout": "whisper_encoder.layers enumerated in named_modules "
                          "order x (q,k,v,out,fc1,fc2)"},
               return_path)
    print(f"  patch-saved -> {return_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
