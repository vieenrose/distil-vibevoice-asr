#!/usr/bin/env python
"""v9 phase B: KL through the FROZEN base decoder, self-labeled by the teacher.

Phase A (scripts/60, prune12) got the probe to zh_kl 0.019 / en_kl 0.085 with
structural margins fully preserved -- and still near-missed every gate band
(golden zh 96.3 vs 97, en 93.3 vs 95, en WER 0.208 vs 0.20, zh_utts 197 vs
250+).  Feature regression has plateaued exactly where scripts/62's rationale
predicts: MSE/cosine weight all 1024 dims equally; the residual that remains
lives in the decoder-relevant directions.  So phase B optimises those
directions directly, with two deliberate differences from scripts/62:

1. FORCING TEXT IS TEACHER-GENERATED, NOT PSEUDO-LABELS.  62 forced on
   ivod_ft_v4 labels -- zh-only (the weak axis is ENGLISH) and the very
   dataset stage 4 flagged.  Here the frozen teacher greedily transcribes each
   training chunk once (cached on disk); KL is then measured along the
   teacher's own trajectory.  No external labels anywhere -- this is exactly
   "make the student's decoder behave like the teacher's".
2. ENGLISH IS UPWEIGHTED.  The pool is sampled from the phase-B manifest
   (AMI entries x4 => ~42% en sampling mass vs 15% in phase A).

Same single-model module-swap trick as 62: one model object, encoder/adaptor
attributes swapped between teacher and student, so the 0.6B decoder is never
duplicated and provably identical across the two forwards.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")


def read_chunk(rng, path, want_s, tgt_sr):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    try:
        info = sf.info(path)
        dur = info.frames / info.samplerate
        if dur <= 0.5:
            return None
        take = min(want_s, dur)
        off = rng.uniform(0, max(0.0, dur - take))
        wav, sr = sf.read(path, start=int(off * info.samplerate),
                          frames=int(take * info.samplerate))
    except Exception:
        return None
    wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
    if wav.size < 800:
        return None
    if sr != tgt_sr:
        g = gcd(sr, tgt_sr)
        wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
    return off, wav


def save_model(out_dir, teacher_dir, model, t_enc, t_ada, enc, ada):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.model.whisper_encoder, model.model.vq_adaptor = t_enc, t_ada
    m = copy.deepcopy(model).float()
    m.model.whisper_encoder = copy.deepcopy(enc).float()
    m.model.vq_adaptor = copy.deepcopy(ada).float()
    m.config.audio_config.encoder_layers = 12
    m.save_pretrained(out)
    for f in Path(teacher_dir).iterdir():
        if f.suffix in {".py", ".jinja"} or f.name.startswith(
                ("tokenizer", "processor", "preprocessor", "generation")):
            if not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    del m
    torch.cuda.empty_cache()
    print(f"saved -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None, help="base snapshot (default: HF cache)")
    ap.add_argument("--init-from", default="models/moss_v9_prune12_base")
    ap.add_argument("--manifest", default="/tmp/claude-1001/train_audio_manifest_b.jsonl")
    ap.add_argument("--pool", type=int, default=500, help="fixed chunk pool size")
    ap.add_argument("--short-ratio", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--feat-weight", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="models/moss_v9b_prune12_kl")
    ap.add_argument("--target-cache", default="/tmp/claude-1001/v9b_targets.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    import glob as _g
    base = args.teacher or _g.glob(
        "/home/luigi/.cache/huggingface/hub/"
        "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/")[0]
    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tok = proc.tokenizer
    fe = proc.feature_extractor
    t_enc, t_ada = model.model.whisper_encoder, model.model.vq_adaptor

    prev = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.init_from), trust_remote_code=True, dtype=torch.float32)
    enc = copy.deepcopy(prev.model.whisper_encoder).to(dev)
    ada = copy.deepcopy(prev.model.vq_adaptor).to(dev)
    del prev
    for p in list(enc.parameters()) + list(ada.parameters()):
        p.requires_grad_(True)
    print(f"init encoder+adaptor from {args.init_from}", flush=True)

    # ---- fixed chunk pool (seeded => cache keys are stable across runs) ----
    paths = []
    with open(args.manifest) as f:
        for line in f:
            p = json.loads(line).get("audio_path")
            if p and Path(p).exists():
                paths.append(p)
    rng = random.Random(7)
    pool = []
    while len(pool) < args.pool:
        is_short = rng.random() < args.short_ratio
        want = rng.uniform(2.0, 10.0) if is_short else 30.0
        got = read_chunk(rng, rng.choice(paths), want, fe.sampling_rate)
        if got is None:
            continue
        off, wav = got
        # seeded rng + deterministic manifest order => stable cache keys
        pool.append({"wav": wav,
                     "key": f"{len(pool)}:{off:.2f}:{len(wav)}"})
    print(f"pool: {len(pool)} chunks", flush=True)

    # ---- teacher self-labels (greedy, cached) ------------------------------
    cache = {}
    cpath = Path(args.target_cache)
    if cpath.exists():
        for line in cpath.open():
            r = json.loads(line)
            cache[r["key"]] = r["text"]
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": DEFAULT_PROMPT}]}]
    prompt_text = proc.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    t0 = time.time()
    with cpath.open("a") as cf:
        for i, c in enumerate(pool):
            if c["key"] in cache:
                continue
            encd = proc(text=prompt_text, audio=[c["wav"]],
                        return_tensors="pt")
            encd = {k: (v.to(dev) if torch.is_tensor(v) else v)
                    for k, v in encd.items()}
            n_prompt = encd["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **encd, do_sample=False,
                    max_new_tokens=int(12 * len(c["wav"]) / fe.sampling_rate) + 64,
                    pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
            cache[c["key"]] = text
            cf.write(json.dumps({"key": c["key"], "text": text},
                                ensure_ascii=False) + "\n")
            cf.flush()
            if (i + 1) % 25 == 0:
                print(f"self-label {i+1}/{len(pool)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"self-labels ready ({time.time()-t0:.0f}s)", flush=True)

    params = list(enc.parameters()) + list(ada.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / args.warmup
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def tokens(e, a, feats):
        h = e(feats, return_dict=True).last_hidden_state
        return a(model.model.time_merge(h))

    hist, step = [], 0
    order = list(range(len(pool)))
    ep_rng = random.Random(13)
    while step < args.steps:
        ep_rng.shuffle(order)
        for idx in order:
            if step >= args.steps:
                break
            c = pool[idx]
            text = cache[c["key"]].strip()
            if not text:
                continue
            step += 1
            full_text = prompt_text + text + tok.eos_token
            try:
                encd = proc(text=full_text, audio=[c["wav"]],
                            max_length=args.max_len, truncation=True,
                            return_tensors="pt")
                n_prompt = proc(text=prompt_text, audio=[c["wav"]],
                                return_tensors="pt")["input_ids"].shape[1]
            except Exception:
                continue
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in encd.items()}
            if batch["input_ids"].shape[1] <= n_prompt + 1:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            feats = batch["input_features"].to(dev, torch.float32)

            model.model.whisper_encoder, model.model.vq_adaptor = t_enc, t_ada
            with torch.no_grad():
                tl = model(**batch).logits[:, n_prompt - 1:-1, :]
                t_tok = tokens(t_enc, t_ada, feats)

            model.model.whisper_encoder, model.model.vq_adaptor = enc, ada
            sl = model(**batch).logits[:, n_prompt - 1:-1, :]
            s_tok = tokens(enc, ada, feats)
            model.model.whisper_encoder, model.model.vq_adaptor = t_enc, t_ada

            kl = F.kl_div(F.log_softmax(sl.float(), -1),
                          F.log_softmax(tl.float(), -1),
                          log_target=True, reduction="batchmean")
            feat = F.mse_loss(s_tok, t_tok) + \
                (1 - F.cosine_similarity(s_tok, t_tok, dim=-1)).mean()
            loss = args.kl_weight * kl + args.feat_weight * feat

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % 100 == 0:
                with torch.no_grad():
                    cs = F.cosine_similarity(s_tok, t_tok, dim=-1).mean().item()
                print(f"step {step:5d}: kl={kl.item():.4f} "
                      f"feat={feat.item():.4f} cos={cs:.4f} "
                      f"lr={lr_at(step):.2e}", flush=True)
                hist.append({"step": step, "kl": kl.item(), "cos": cs})
            if step % args.save_every == 0 or step == args.steps:
                save_model(ROOT / args.out, base, model, t_enc, t_ada, enc, ada)
                (ROOT / args.out / "kl_history.json").write_text(
                    json.dumps(hist, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
