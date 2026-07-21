#!/usr/bin/env python
"""Distil MOSS-TD's Whisper-medium encoder into a lighter one, near-losslessly.

Goal (2026-07-20): lighter encoder at ~no quality cost on BOTH long and short
audio. The encoder is 56% of wall-clock (measured on Pi4 AND x86) and is the
only part of the model never pruned.

Target = the teacher's VQAdaptor OUTPUT, frame by frame. That is the exact
1024-d token stream the Qwen3 decoder consumes, so if the student matches it the
decoder cannot tell the encoder was replaced -- and whatever speaker information
the teacher encodes is inside the target, so diarization survives by
construction rather than by hope. No transcripts are needed: any audio works.

TWO THINGS THIS SCRIPT DOES THAT A NAIVE DISTILL WOULD GET WRONG
----------------------------------------------------------------
1. MIXED CHUNK DISTRIBUTION. Whisper pads every clip to a 30 s chunk. A short
   2-8 s utterance therefore presents the encoder with a mostly-SILENT chunk --
   a completely different input distribution from a dense 30 s window of
   continuous parliamentary speech. Training on long windows alone would quietly
   wreck short-utterance accuracy (and vice versa). We sample both, at a
   controllable ratio, every batch.
2. SEPARATE HELD-OUT REPORTING. Cosine is reported for the short-style and
   long-style held-out sets INDEPENDENTLY. A single averaged number can hide a
   regression in exactly one of the two regimes we promised not to break.

The saved artifact is a complete MOSS model directory with the encoder and
adaptor swapped and the decoder copied verbatim, so scripts/45, scripts/57 and
the GGUF conversion all work on it unchanged.

Whatever comes out of here still needs the CE-only QAT pass afterwards (v7
lesson: QAT and distillation-KL are antagonistic -- keep the QAT objective
simple, and never gate on loss magnitude under fake-quant).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# whisper-small encoder geometry
SMALL = dict(d_model=768, encoder_layers=12, encoder_attention_heads=12,
             encoder_ffn_dim=3072)


# ----------------------------------------------------------------- data -----
class ChunkSampler:
    """Yields (B,80,3000) input_features mixing long-dense and short-padded."""

    def __init__(self, long_paths, short_paths, feat_ext, seed=0,
                 short_ratio=0.35, short_range=(2.0, 10.0), chunk_s=30.0):
        self.long_paths = long_paths
        self.short_paths = short_paths or long_paths
        self.fe = feat_ext
        self.rng = random.Random(seed)
        self.short_ratio = short_ratio
        self.short_range = short_range
        self.chunk_s = chunk_s

    def _read(self, path, want_s):
        import soundfile as sf
        from math import gcd
        from scipy.signal import resample_poly
        try:
            info = sf.info(path)
            dur = info.frames / info.samplerate
            if dur <= 0.5:
                return None
            take = min(want_s, dur)
            off = self.rng.uniform(0, max(0.0, dur - take))
            wav, sr = sf.read(path, start=int(off * info.samplerate),
                              frames=int(take * info.samplerate))
        except Exception:
            return None
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
        if wav.size < 800:
            return None
        tgt = self.fe.sampling_rate
        if sr != tgt:
            g = gcd(sr, tgt)
            wav = resample_poly(wav, tgt // g, sr // g).astype(np.float32)
        return wav

    def batch(self, n):
        feats, kinds = [], []
        tries = 0
        while len(feats) < n and tries < n * 8:
            tries += 1
            is_short = self.rng.random() < self.short_ratio
            if is_short:
                w = self._read(self.rng.choice(self.short_paths),
                               self.rng.uniform(*self.short_range))
            else:
                w = self._read(self.rng.choice(self.long_paths), self.chunk_s)
            if w is None:
                continue
            feats.append(self.fe(w, sampling_rate=self.fe.sampling_rate,
                                 return_tensors="pt").input_features[0])
            kinds.append("short" if is_short else "long")
        if not feats:
            return None, None
        return torch.stack(feats), kinds

    def fixed(self, n, kind):
        """Deterministic held-out batch of one kind."""
        keep_ratio, keep_seed = self.short_ratio, self.rng
        self.short_ratio = 1.0 if kind == "short" else 0.0
        self.rng = random.Random(999 if kind == "short" else 888)
        f, _ = self.batch(n)
        self.short_ratio, self.rng = keep_ratio, keep_seed
        return f


# -------------------------------------------------------------- students ----
def build_student(kind, teacher):
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    from transformers import WhisperForConditionalGeneration

    t_enc = teacher.model.whisper_encoder
    t_ada = teacher.model.vq_adaptor

    if kind == "prune12":
        cfg = copy.deepcopy(t_enc.config)
        cfg.encoder_layers = 12
        enc = WhisperEncoder(cfg)
        sd, keep, new_sd = t_enc.state_dict(), list(range(0, 24, 2)), {}
        for k, v in sd.items():
            if k.startswith("layers."):
                i = int(k.split(".")[1])
                if i in keep:
                    new_sd[k.replace(f"layers.{i}.",
                                     f"layers.{keep.index(i)}.", 1)] = v
            else:
                new_sd[k] = v
        enc.load_state_dict(new_sd, strict=False)
        ada = copy.deepcopy(t_ada)
        d_model = cfg.d_model
    elif kind == "small":
        w = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", dtype=torch.float32)
        enc = w.model.encoder
        del w
        ada = type(t_ada)(input_dim=SMALL["d_model"] * 4, hidden_size=1024,
                          norm_eps=1e-6)
        with torch.no_grad():
            ada.layers[2].load_state_dict(t_ada.layers[2].state_dict())
            ada.layers[3].load_state_dict(t_ada.layers[3].state_dict())
        d_model = SMALL["d_model"]
    else:
        raise ValueError(kind)
    return enc.float(), ada.float(), d_model


def student_tokens(enc, ada, feats, merge=4):
    h = enc(feats, return_dict=True).last_hidden_state
    B, T, D = h.shape
    Tt = (T // merge) * merge
    return ada(h[:, :Tt, :].reshape(B, Tt // merge, D * merge))


def teacher_tokens(teacher, feats):
    with torch.no_grad():
        h = teacher.model.whisper_encoder(feats, return_dict=True).last_hidden_state
        return teacher.model.vq_adaptor(teacher.model.time_merge(h))


# ----------------------------------------------------------------- save -----
def save_model(out_dir, teacher_dir, teacher, enc, ada, kind, d_model):
    """Write a complete MOSS dir with the encoder/adaptor swapped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(teacher).float()
    # deepcopy the student too: assigning the live objects would alias them into
    # `model`, and the .to(bfloat16) below would convert the TRAINING encoder in
    # place -- fp32 activations then hit bf16 weights on the next forward.
    model.model.whisper_encoder = copy.deepcopy(enc).float()
    model.model.vq_adaptor = copy.deepcopy(ada).float()
    cfg = model.config
    if kind == "small":
        for k, v in SMALL.items():
            setattr(cfg.audio_config, k, v)
        cfg.adaptor_input_dim = d_model * 4
    else:
        cfg.audio_config.encoder_layers = 12
    model.config = cfg
    model.to(torch.bfloat16).save_pretrained(out)
    # trust_remote_code + processor/tokenizer assets travel with the weights
    for f in Path(teacher_dir).iterdir():
        if f.suffix in {".py", ".jinja"} or f.name.startswith(
                ("tokenizer", "processor", "preprocessor", "generation")):
            if not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    print(f"saved -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/moss_ft_zhtw_v7")
    ap.add_argument("--student", default="small", choices=["small", "prune12"])
    ap.add_argument("--long-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--short-ratio", type=float, default=0.35)
    ap.add_argument("--n-wavs", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="models/moss_v8_encsmall")
    ap.add_argument("--init-from", default=None,
                    help="resume: load encoder+adaptor from a saved swap dir")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    dev = torch.device(args.device)
    tdir = str(ROOT / args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        tdir, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(tdir, trust_remote_code=True)
    fe = proc.feature_extractor

    paths = []
    with open(ROOT / args.long_manifest) as f:
        for line in f:
            try:
                p = json.loads(line).get("audio_path")
            except Exception:
                continue
            if p and Path(p).exists():
                paths.append(p)
    random.Random(0).shuffle(paths)
    paths = paths[:args.n_wavs]
    n_hold = max(8, len(paths) // 20)
    hold, train = paths[:n_hold], paths[n_hold:]
    print(f"audio: {len(train)} train / {len(hold)} held-out", flush=True)

    tr = ChunkSampler(train, train, fe, seed=7, short_ratio=args.short_ratio)
    ho = ChunkSampler(hold, hold, fe, seed=11, short_ratio=args.short_ratio)
    ev_short = ho.fixed(6, "short").to(dev)
    ev_long = ho.fixed(6, "long").to(dev)

    enc, ada, d_model = build_student(args.student, teacher)
    if args.init_from:
        prev = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.init_from), trust_remote_code=True,
            dtype=torch.float32)
        enc.load_state_dict(prev.model.whisper_encoder.state_dict())
        ada.load_state_dict(prev.model.vq_adaptor.state_dict())
        del prev
        print(f"resumed encoder+adaptor from {args.init_from}", flush=True)
    enc, ada = enc.to(dev), ada.to(dev)
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"student={args.student} encoder={n_par/1e6:.1f}M (teacher 307M)",
          flush=True)

    params = list(enc.parameters()) + list(ada.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    def ev(feats):
        enc.eval(); ada.eval()
        with torch.no_grad():
            t = teacher_tokens(teacher, feats)
            p = student_tokens(enc, ada, feats)
            cos = F.cosine_similarity(p, t, dim=-1).mean().item()
            rel = ((p - t).norm() / t.norm()).item()
        enc.train(); ada.train()
        return cos, rel

    hist = []
    cs, rs = ev(ev_short); cl, rl = ev(ev_long)
    print(f"step     0: short_cos={cs:.4f} long_cos={cl:.4f}", flush=True)
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        feats, _ = tr.batch(args.batch)
        if feats is None:
            continue
        feats = feats.to(dev)
        tgt = teacher_tokens(teacher, feats)
        pred = student_tokens(enc, ada, feats)
        loss = F.mse_loss(pred, tgt) + \
            (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.eval_every == 0:
            cs, rs = ev(ev_short); cl, rl = ev(ev_long)
            print(f"step {step:5d}: loss={loss.item():.4f} lr={lr_at(step):.2e} "
                  f"short_cos={cs:.4f} (rel {rs:.3f})  "
                  f"long_cos={cl:.4f} (rel {rl:.3f})", flush=True)
            hist.append({"step": step, "short_cos": cs, "long_cos": cl,
                         "short_rel": rs, "long_rel": rl})
        if step % args.save_every == 0 or step == args.steps:
            save_model(ROOT / args.out, tdir, teacher, enc, ada,
                       args.student, d_model)
            (ROOT / args.out / "distill_history.json").write_text(
                json.dumps(hist, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
