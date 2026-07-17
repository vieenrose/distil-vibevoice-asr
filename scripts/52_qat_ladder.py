"""QAT for near-lossless int4: fine-tune v4 with STE int4 fake-quant on the
decoder Linears that become MatMulNBits in the q4 export.

Weights adapt to the exact 4-bit block-32 symmetric rounding, so the real q4
export loses far less than post-training RTN. Optionally keeps the most
q4-sensitive groups (from scripts/48) in full precision -> int8 at export
(mixed precision). Same SFT objective as v4 (CE on assistant tokens).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def target_text(segments) -> str:
    return "".join(f"[{s['start']:.2f}][S{int(s['speaker'])+1:02d}]{s['text']}"
                   f"[{s['end']:.2f}]" for s in segments)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v5_kl_qat_fixed")
    ap.add_argument("--patch-save", action="store_true",
                    help="save only the QAT-trained Linear weights as a patch")
    ap.add_argument("--ffn-only", action="store_true",
                    help="fake-quant only gate/up/down projections (mixed "
                         "precision: attention/embeds stay at export default)")
    ap.add_argument("--bits", type=int, default=3,
                    help="fake-quant bit width: 4 / 3 / 2 (2 = block ternary)")
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--max-audio-s", type=float, default=300.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--exclude-sensitivity", default="data/q4_sensitivity.json")
    ap.add_argument("--keep-int8-top", type=int, default=0,
                    help="keep the N most q4-sensitive groups OUT of QAT "
                         "(they stay int8 at export)")
    ap.add_argument("--out", default="models/moss_ft_zhtw_v4_qat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anneal", default="",
                    help="FFN bit anneal, e.g. '4:450,3:900' = bits 4 until "
                         "step 450, 3 until 900, then --bits. Attention "
                         "targets stay at 4.")
    ap.add_argument("--kl-w", type=float, default=0.0,
                    help="weight of self-distill KL (teacher = same weights, "
                         "fake-quant disabled)")
    ap.add_argument("--spare-top", type=int, default=0,
                    help="keep FFN of the N most q4-sensitive layers at 4 bits")
    ap.add_argument("--freeze-rest", action="store_true",
                    help="train only the wrapped QAT latents (required for "
                         "--patch-save to capture the learning)")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.data.augment import augment_wav
    from distil_vibevoice.quant.fakequant import wrap_decoder_linears

    dev = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.model), trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(str(ROOT / args.model),
                                         trust_remote_code=True)

    # decide which groups to keep at int8 (exclude from fake-quant)
    exclude_frag = set()
    sp = ROOT / args.exclude_sensitivity
    if args.keep_int8_top > 0 and sp.exists():
        ranked = json.loads(sp.read_text())["ranked"]
        keep = [k for k, _ in ranked[:args.keep_int8_top]]
        exclude_frag = set(keep)
        print(f"keeping int8 (excluded from QAT): {keep}", flush=True)

    def excluded(full_name: str) -> bool:
        if "lm_head" in full_name and "lm_head" in exclude_frag:
            return True
        import re
        m = re.search(r"layers\.(\d+)\.", full_name)
        return bool(m) and f"layer{int(m.group(1)):02d}" in exclude_frag

    all_names = [n for n, _ in model.named_modules()]
    excl = {n for n in all_names if excluded(n)}
    targets = (("gate_proj", "up_proj", "down_proj") if args.ffn_only
               else ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj", "lm_head"))
    wrap_bits = 4 if args.anneal else args.bits
    wrapped = wrap_decoder_linears(model, targets=targets, exclude=excl, bits=wrap_bits)
    print(f"QAT wrapping {len(wrapped)} Linear layers (bits={wrap_bits})", flush=True)

    # Anneal plan: FFN modules step down 4 -> 3 -> --bits on a schedule while
    # attention/lm_head stay at 4 (their deployment precision). The N most
    # q4-sensitive layers keep their FFN at 4 bits permanently.
    anneal_sched, ffn_mods = [], []
    if args.anneal:
        from distil_vibevoice.quant.fakequant import QATLinear as _QL
        for part in args.anneal.split(","):
            b, until = part.split(":")
            anneal_sched.append((int(until), int(b)))
        anneal_sched.sort()
        spared = set()
        if args.spare_top:
            sens = json.loads((ROOT / args.exclude_sensitivity).read_text())["sensitivity"]
            spared = {k for k, _ in sorted(sens.items(), key=lambda kv: -kv[1])
                      [:args.spare_top]}
        import re as _re
        for n, m in model.named_modules():
            if isinstance(m, _QL) and any(
                    t in n.rsplit(".", 1)[-1] for t in ("gate_proj", "up_proj", "down_proj")):
                lm = _re.search(r"layers\.(\d+)\.", n)
                if lm and f"layer{int(lm.group(1)):02d}" in spared:
                    continue
                ffn_mods.append(m)
        print(f"anneal {args.anneal} -> {args.bits} on {len(ffn_mods)} FFN mods "
              f"(spared {len(spared)} layers)", flush=True)

    def ffn_bits_at(step):
        for until, b in anneal_sched:
            if step < until:
                return b
        return args.bits

    if args.freeze_rest:
        # Train ONLY the wrapped latents. Otherwise the unconstrained params
        # absorb the quantization error, the latents never adapt to the
        # lattice, and --patch-save (which persists just the wrapped tensors)
        # ships a model indistinguishable from PTQ.
        from distil_vibevoice.quant.fakequant import QATLinear
        for p in model.parameters():
            p.requires_grad_(False)
        n_train = 0
        for m in model.modules():
            if isinstance(m, QATLinear):
                m.lin.weight.requires_grad_(True)
                n_train += 1
        print(f"freeze-rest: training only {n_train} wrapped latents", flush=True)

    model.gradient_checkpointing_enable()
    if args.freeze_rest:
        model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    tts = []
    for mp in args.tts_manifests:
        p = ROOT / mp
        if p.exists():
            tts += [json.loads(l) for l in p.open()]
    ivod = ([json.loads(l) for l in (ROOT / args.ivod_manifest).open()]
            if (ROOT / args.ivod_manifest).exists() else [])
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rng.shuffle(tts)
    rng.shuffle(ivod)
    tgt_sr = proc.feature_extractor.sampling_rate

    def window_sample(rec):
        dur = float(rec.get("duration") or max(s["end"] for s in rec["segments"]))
        off = 0.0 if dur <= args.max_audio_s else rng.uniform(0, dur - args.max_audio_s)
        inside = [s for s in rec["segments"]
                  if s["start"] >= off and s["end"] <= off + args.max_audio_s]
        if not inside:
            return None, None
        spk_map, segs = {}, []
        for s in sorted(inside, key=lambda x: x["start"]):
            if s["speaker"] not in spk_map:
                spk_map[s["speaker"]] = len(spk_map) + 1
            segs.append({"start": s["start"] - off, "end": s["end"] - off,
                         "speaker": spk_map[s["speaker"]] - 1, "text": s["text"]})
        return off, segs

    losses, ti, ii = [], 0, 0
    for step in range(args.steps):
        use_ivod = ivod and rng.random() < args.p_ivod
        rec = (ivod[ii % len(ivod)] if use_ivod else tts[ti % len(tts)])
        if use_ivod:
            ii += 1
        else:
            ti += 1
        off, segs = window_sample(rec)
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
        if not use_ivod and rng.random() < 0.3:
            wav = augment_wav(wav, sr, rir_dir="data/aug/RIRS_NOISES",
                              musan_dir="data/aug/musan", rng=np_rng)
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        ptext = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=ptext + target_text(segs) + proc.tokenizer.eos_token,
                   audio=[wav], max_length=args.max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        npf = proc(text=ptext, audio=[wav], max_length=args.max_len,
                   return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :npf] = -100
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
        batch["labels"] = labels
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if ffn_mods:
            fb = ffn_bits_at(step)
            for m in ffn_mods:
                m.bits = fb
        if args.kl_w > 0:
            from distil_vibevoice.quant.fakequant import set_fakequant
            with torch.no_grad():
                set_fakequant(model, False)
                t_logits = model(**{k: v for k, v in batch.items()
                                    if k != "labels"}).logits
                set_fakequant(model, True)
        out = model(**batch)
        loss = out.loss
        if args.kl_w > 0:
            mask = batch["labels"][:, 1:] != -100
            s = out.logits[:, :-1][mask].float()
            t = t_logits[:, :-1][mask].float()
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(s, -1), torch.softmax(t, -1),
                reduction="batchmean")
            loss = loss + args.kl_w * kl
            del t_logits
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 20 == 0:
            fb = f" ffn_bits={ffn_bits_at(step)}" if ffn_mods else ""
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e}{fb}", flush=True)

    # unwrap: fold nothing — save the raw (full-precision) weights that were
    # trained to be int4-robust; the ONNX q4 export applies the real rounding.
    import torch.nn as nn
    for name, module in list(model.named_modules()):
        for cn, child in list(module.named_children()):
            if child.__class__.__name__ == "QATLinear":
                setattr(module, cn, child.lin)
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    # Disk-frugal fallback: QAT only mutates the wrapped Linears, so persist
    # JUST those tensors (~0.5 GB for FFN-only) and overlay them on the base
    # checkpoint at export time. Full save_pretrained needs ~1.9 GB and died
    # on disk quota once already.
    if getattr(args, "patch_save", False):
        patch = {n + ".weight": m.weight.detach().to(torch.bfloat16).cpu()
                 for n, m in model.named_modules()
                 if m.__class__.__name__ == "Linear" and
                    any(t in n.rsplit(".", 1)[-1] for t in targets)}
        torch.save({"base": args.model, "bits": args.bits, "tensors": patch},
                   outdir / "qat_patch.pt")
        print(f"patch-saved {len(patch)} tensors -> {outdir}/qat_patch.pt", flush=True)
    else:
        model.save_pretrained(outdir)
        proc.save_pretrained(outdir)
    print(f"\nQAT loss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f} | -> {outdir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
