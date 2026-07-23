#!/usr/bin/env python
"""Decoder q2_K QAT: q4 -> 2-bit decoder weights at (gated) teacher equality.

Stage-0 PTQ probes (2026-07-22) set the frame: ternary PTQ destroys the
decoder outright (agree 1.4-1.9% — capacity wall, retry pointless), but a
q2_K-proxy (4-level asymmetric, block-16 scale+min) lands at agree 88.1 zh /
87.3 en, KL 0.35/0.45 — classic repairable quant error, the same class QAT
closed for q4 in the v5/v7 era.

Every hard-won QAT lesson applied:
  * FREEZE-REST (scripts/52 lesson): only the wrapped decoder-linear latents
    train. Training everything lets unquantized params absorb the error and
    the deployed model is garbage-PTQ.
  * CE-ONLY (v7 lesson): KL fights fake-quant; self-KL diverges.
  * lr 1e-4 (1e-5 plateaus), cosine, grad-accum 8 (v9b batch=1 noise lesson).
  * token_embd untouched (the proven segmentation-collapse tensor), encoder/
    adaptor untouched (they are not the target and freeze-rest excludes them).
  * Targets: MOSS-TD BASE f32 self-labels (transcript+speaker-tags+timestamps
    jointly; per user directive pseudo-labels must come from MOSS-TD, never
    the VibeVoice/WhisperX-era pipelines), structural tokens upweighted 4x
    ('['/digits/Sxx -- protects diarization + marker cadence under QAT).
  * Deployment gate is the REAL artifact: save f32 (latents), engine-quantize
    decoder tensors to real q2_K, splice, then scripts/86 battery + unseen-
    meeting A/B. The in-training proxy underestimates real q2_K error (whose
    block scales are themselves 4-bit), so only the real-GGUF battery counts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")


def fq_q2k(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    """q2_K-proxy fake-quant: 4-level asymmetric per block-16 (scale+min)."""
    out_f, in_f = w.shape
    pad = (-in_f) % block
    wv = F.pad(w, (0, pad)).view(out_f, -1, block)
    mn = wv.amin(-1, keepdim=True)
    scale = ((wv.amax(-1, keepdim=True) - mn) / 3).clamp(min=1e-8)
    q = torch.clamp(torch.round((wv - mn) / scale), 0, 3)
    return (q * scale + mn).view(out_f, -1)[:, :in_f]


class Q2KLinear(nn.Module):
    """STE fake-quant wrapper; latent weight is the trainable parameter."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = (nn.Parameter(lin.bias.detach().clone())
                     if lin.bias is not None else None)

    def forward(self, x):
        w = self.weight
        wq = w + (fq_q2k(w) - w).detach()   # STE
        return F.linear(x, wq, self.bias)

    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.weight.shape[1], self.weight.shape[0],
                        bias=self.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.weight)   # latents; real quant at GGUF time
            if self.bias is not None:
                lin.bias.copy_(self.bias)
        return lin


def wrap_decoder(model):
    wrapped = []
    for name, mod in model.named_modules():
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if (isinstance(child, nn.Linear) and ".layers." in full
                    and "whisper" not in full and "vq_adaptor" not in full):
                setattr(mod, cname, Q2KLinear(child))
                wrapped.append(full)
    return wrapped


def unwrap_decoder(model):
    for name, mod in model.named_modules():
        for cname, child in list(mod.named_children()):
            if isinstance(child, Q2KLinear):
                setattr(mod, cname, child.to_linear())


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


def structural_ids(tok):
    ids = set()
    for s in list("[]0123456789.") + [f"S{i:02d}" for i in range(1, 12)]:
        for i in tok(s, add_special_tokens=False).input_ids:
            ids.add(i)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--manifest", default="/tmp/claude-1001/train_audio_manifest_b.jsonl")
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--short-ratio", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--struct-weight", type=float, default=4.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="models/moss_q2k_decoder_qat")
    ap.add_argument("--target-cache", default="/tmp/claude-1001/v9b_targets.jsonl")
    args = ap.parse_args()

    # torchrun DDP: each rank runs micro-batches on its own GPU; gradients
    # all-reduce, so one optimizer step == accum*world micro-batches.
    import os as _os
    ddp = "RANK" in _os.environ
    rank = int(_os.environ.get("RANK", "0"))
    world = int(_os.environ.get("WORLD_SIZE", "1"))
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        args.device = f"cuda:{rank}"
    is_main = rank == 0

    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    import glob as _g
    base = args.teacher or _g.glob(
        "/home/luigi/.cache/huggingface/hub/"
        "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/")[0]
    dev = torch.device(args.device)
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tok = proc.tokenizer
    fe = proc.feature_extractor

    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, dtype=torch.float32).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = wrap_decoder(model)
    n_lat = 0
    for name, mod in model.named_modules():
        if isinstance(mod, Q2KLinear):
            mod.weight.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)
            n_lat += mod.weight.numel()
    if is_main:
        print(f"QAT: wrapped {len(wrapped)} decoder linears "
              f"({n_lat/1e6:.0f}M latents trainable, rest FROZEN), "
              f"world={world}", flush=True)
    model.train()
    net = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[rank])

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
        p = rng.choice(paths)
        got = read_chunk(rng, p, want, fe.sampling_rate)
        if got is None:
            continue
        off, wav = got
        pool.append({"wav": wav, "key": f"{len(pool)}:{off:.2f}:{len(wav)}"})
    print(f"pool: {len(pool)} chunks", flush=True)

    cache = {}
    cpath = Path(args.target_cache)
    if cpath.exists():
        for line in cpath.open():
            r = json.loads(line)
            cache[r["key"]] = r["text"]
    missing = [c for c in pool if c["key"] not in cache]
    assert not missing, f"{len(missing)} chunks lack self-labels (expected v9b cache reuse)"

    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": DEFAULT_PROMPT}]}]
    prompt_text = proc.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    sids = structural_ids(tok)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def save(step):
        if not is_main:
            return
        import copy as _copy
        snap = _copy.deepcopy(model).float()
        unwrap_decoder(snap)
        out = ROOT / args.out
        snap.save_pretrained(out)
        for f in Path(base).iterdir():
            if f.suffix in {".py", ".jinja"} or f.name.startswith(
                    ("tokenizer", "processor", "preprocessor", "generation")):
                if not (out / f.name).exists():
                    shutil.copy2(f, out / f.name)
        del snap
        torch.cuda.empty_cache()
        print(f"step {step}: saved (latents) -> {out}", flush=True)

    order = list(range(len(pool)))
    ep_rng = random.Random(13)
    step = micro = 0
    running = 0.0
    t0 = time.time()
    while step < args.steps:
        ep_rng.shuffle(order)
        for idx in order[rank::world]:
            if step >= args.steps:
                break
            c = pool[idx]
            text = cache[c["key"]].strip()
            if not text:
                continue
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
            out = net(**batch)
            logits = out.logits[0, n_prompt - 1:-1, :]
            targets = batch["input_ids"][0, n_prompt:]
            w = torch.ones_like(targets, dtype=torch.float32)
            for i, t in enumerate(targets.tolist()):
                if t in sids:
                    w[i] = args.struct_weight
            ce = F.cross_entropy(logits.float(), targets, reduction="none")
            loss = (ce * w).sum() / w.sum() / args.accum
            loss.backward()
            running += loss.item() * args.accum
            micro += 1
            if micro % args.accum == 0:
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0 and is_main:
                    print(f"step {step:5d}: ce={running/(50*args.accum):.4f} "
                          f"lr={lr_at(step):.2e} ({time.time()-t0:.0f}s)",
                          flush=True)
                    running = 0.0
                if step % args.save_every == 0 or step == args.steps:
                    save(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
