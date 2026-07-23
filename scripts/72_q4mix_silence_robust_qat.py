#!/usr/bin/env python
"""q4mix-v2 QAT: make the deployed q4 decoder robust to near-silent input,
at same-or-better transcription accuracy than the current deployed q4mix.

ROOT CAUSE (diagnosed 2026-07-23, scripts/silence_robustness_probe.py, using
the REAL dequantized q4_K weights from the deployed GGUF -- not a proxy):
on near-silent audio the f32 teacher's decision to stop (EOS) at the very
first decode step is CORRECT but only barely confident (margin +1.35 logits
over the runner-up, '['). On real speech the analogous first-step decision
wins by 15-17 logits -- a landslide, immune to any realistic perturbation.
Quantizing the decoder's 196 linears (q4_K) introduces noise on the order of
single-digit-tenths of a logit per matrix, which cannot touch the confident
real-speech margin but is enough, compounded across 28 layers, to flip the
already-thin silence margin. Once EOS loses that one coin-flip the model has
no trained recovery (it has never seen "I emitted a spurious marker with
nothing to say") and free-runs a timestamp/tag loop to the token budget.
Critically: token_embd/lm_head stayed FULL F32 in that diagnostic (more
precise than deployed q4mix's f16) and the fragility appeared anyway -- the
fragile computation is in the attention+FFN stack itself, not the output
projection, so protecting token_embd (which q4mix already does) does not fix
this.

FIX: QAT the decoder with FAKE Q4 quantization in the forward pass (so the
trained latents literally experience the quantization noise during training
and can widen the margin under it), on a training pool that MIXES:
  (a) normal audio + self-labeled transcripts (unchanged from the base q4mix
      lineage -- this is what preserves/improves ordinary transcription
      accuracy), and
  (b) silence-negative chunks, energy-flagged then F32-TEACHER-CONFIRMED
      (scripts/scan_silence_chunks.py: kept only if the real f32 model
      itself emits EOS within 2 tokens on that exact chunk -- the target is
      grounded in actual model behavior, not an assumption), trained with a
      heavily upweighted CE loss on the single EOS decision -- directly
      attacking the exact margin the diagnostic identified as fragile.

Every hard-won QAT lesson from this project applied:
  * FREEZE-REST: only wrapped decoder-linear latents train (unquantized
    params must not be trainable, or they absorb the error and the deployed
    model is garbage-PTQ in disguise).
  * CE-ONLY (v7 lesson): no KL term (KL fights fake-quant / self-KL diverges).
  * lr 1e-4 (1e-5 plateaus), cosine, grad-accum (batch=1 noise lesson).
  * token_embd/lm_head and encoder/adaptor untouched -- not the target, and
    the diagnostic proved they aren't where the fragility lives.
  * Fake-quant is the project's VALIDATED block=32 symmetric int4 scheme
    (src/distil_vibevoice/quant/fakequant.py -- the same recipe behind the
    historical "q4-robust" QAT). It is a proxy for the real deployed Q4_K
    superblock format; the training-time gap is why the ONLY gate that
    counts is the real-GGUF battery (scripts/86 + unseen-meeting A/B),
    exactly as this project has done for every quant decision to date.
  * Targets: MOSS-TD BASE f32 self-labels (never VibeVoice/WhisperX-era
    pseudo-labels), structural tokens upweighted 4x.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")
sys.path.insert(0, str(ROOT / "src"))


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


def read_fixed_chunk(path, off_s, dur_s, tgt_sr):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    info = sf.info(path)
    sr = info.samplerate
    wav, _ = sf.read(path, start=int(off_s * sr), frames=int(dur_s * sr))
    wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
    if sr != tgt_sr:
        g = gcd(sr, tgt_sr)
        wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
    return wav


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
    ap.add_argument("--silence-manifest", default="/tmp/claude-1001/silence_train_manifest.jsonl")
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--short-ratio", type=float, default=0.35)
    ap.add_argument("--silence-frac", type=float, default=0.20,
                    help="fraction of TRAINING STEPS spent on silence-negative examples")
    ap.add_argument("--quiet-manifest", default="/tmp/claude-1001/quiet_speech_manifest.jsonl")
    ap.add_argument("--quiet-frac", type=float, default=0.12,
                    help="fraction of steps on LOW-AMPLITUDE SPEECH positives "
                         "(teacher-labeled attenuated audio -- keeps the model "
                         "sensitive to quiet speech while silence-negatives "
                         "teach it to stop on unvoiced/near-silent input)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--struct-weight", type=float, default=4.0)
    ap.add_argument("--silence-weight", type=float, default=6.0,
                    help="loss weight on the silence-negative EOS decision -- "
                         "the exact margin the diagnostic found fragile")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="models/moss_q4mix_v2_qat")
    ap.add_argument("--target-cache", default="/tmp/claude-1001/v9b_targets.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    from distil_vibevoice.quant.fakequant import wrap_decoder_linears, QATLinear

    import glob as _g
    base = args.teacher or _g.glob(
        "/home/luigi/.cache/huggingface/hub/"
        "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/")[0]
    dev = torch.device(args.device)
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tok = proc.tokenizer
    fe = proc.feature_extractor
    eos_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, dtype=torch.float32).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    # Decoder-only, q/k/v/o + FFN -- explicitly EXCLUDES lm_head (tied to
    # token_embd, which stays f16 in q4mix and was proven NOT the fragile
    # tensor by the diagnostic). Whisper encoder untouched: freeze-rest
    # already skips it (name-fragment match only hits q/k/v/o/gate/up/down
    # in ANY module including the encoder -- so restrict explicitly).
    targets = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
    wrapped = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if (isinstance(child, torch.nn.Linear) and any(t in cname for t in targets)
                    and ".layers." in full and "whisper" not in full):
                setattr(mod, cname, QATLinear(child, block=32, bits=4))
                wrapped.append(full)
    n_lat = sum(m.lin.weight.numel() for m in model.modules() if isinstance(m, QATLinear))
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.lin.weight.requires_grad_(True)
            if m.lin.bias is not None:
                m.lin.bias.requires_grad_(True)
    print(f"QAT: wrapped {len(wrapped)} decoder linears ({n_lat/1e6:.0f}M latents "
          f"trainable, rest FROZEN, q4 block=32 symmetric fake-quant)", flush=True)

    # ENCODER fake-quant, FROZEN (noise injection only, nothing trains): the
    # deployed q4mix quantizes the whisper encoder to q4_K too, and the
    # step-500 real-GGUF smoke showed the silence margin trained against f32
    # encoder outputs did not survive the deployed q4 encoder's output shift.
    # Wrapping the encoder linears (requires_grad stays False) makes every
    # training forward see deployment-like encoder noise.
    enc_targets = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    n_enc = 0
    for name, mod in list(model.named_modules()):
        if "whisper" not in name:
            continue
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and any(t in cname for t in enc_targets):
                setattr(mod, cname, QATLinear(child, block=32, bits=4))
                n_enc += 1
    print(f"QAT: wrapped {n_enc} ENCODER linears (frozen, fake-quant noise "
          f"only -- deployment-like inputs to the decoder)", flush=True)
    model.train()

    # ---- normal pool (same recipe/seed as the base q4mix decoder QAT lineage) --
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
    print(f"normal pool: {len(pool)} chunks", flush=True)

    cache = {}
    cpath = Path(args.target_cache)
    if cpath.exists():
        for line in cpath.open():
            r = json.loads(line)
            cache[r["key"]] = r["text"]
    missing = [c for c in pool if c["key"] not in cache]
    assert not missing, (f"{len(missing)} chunks lack cached self-labels -- "
                         f"expected full reuse of {args.target_cache}")

    # ---- silence-negative pool (F32-teacher-confirmed, scan_silence_chunks.py) -
    sil_pool = []
    if Path(args.silence_manifest).exists():
        with open(args.silence_manifest) as f:
            for line in f:
                r = json.loads(line)
                sil_pool.append(r)
    print(f"silence-negative pool: {len(sil_pool)} confirmed chunks", flush=True)
    assert sil_pool, "no confirmed silence chunks -- run scan_silence_chunks.py first"

    quiet_pool = []
    for mf in (args.quiet_manifest, "/tmp/claude-1001/patience_manifest.jsonl"):
        if Path(mf).exists():
            with open(mf) as f:
                for line in f:
                    quiet_pool.append(json.loads(line))
    print(f"quiet-speech positive pool: {len(quiet_pool)} teacher-labeled chunks",
          flush=True)
    assert quiet_pool, "no quiet-speech positives -- run gen_quiet_speech.py first"

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

    def save(step, tag=""):
        from distil_vibevoice.quant.fakequant import set_fakequant
        import copy as _copy
        set_fakequant(model, False)
        snap = _copy.deepcopy(model).float()
        set_fakequant(model, True)
        for name, mod in list(snap.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, QATLinear):
                    setattr(mod, cname, child.lin)
        out = ROOT / args.out / f"step_{step}"
        snap.save_pretrained(out)
        for f in Path(base).iterdir():
            if f.suffix in {".py", ".jinja"} or f.name.startswith(
                    ("tokenizer", "processor", "preprocessor", "generation")):
                if not (out / f.name).exists():
                    shutil.copy2(f, out / f.name)
        del snap
        torch.cuda.empty_cache()
        print(f"step {step}{tag}: saved (fp32 latents, q4-QAT'd) -> {out}", flush=True)

    def ce_example(wav, text):
        """Shared full-transcript CE with structural-token upweighting."""
        full_text = prompt_text + text + tok.eos_token
        try:
            encd = proc(text=full_text, audio=[wav],
                        max_length=args.max_len, truncation=True,
                        return_tensors="pt")
            n_prompt = proc(text=prompt_text, audio=[wav],
                            return_tensors="pt")["input_ids"].shape[1]
        except Exception:
            return None
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in encd.items()}
        if batch["input_ids"].shape[1] <= n_prompt + 1:
            return None
        out = model(**batch)
        logits = out.logits[0, n_prompt - 1:-1, :]
        targets_ = batch["input_ids"][0, n_prompt:]
        w = torch.ones_like(targets_, dtype=torch.float32)
        for i, t in enumerate(targets_.tolist()):
            if t in sids:
                w[i] = args.struct_weight
        ce = F.cross_entropy(logits.float(), targets_, reduction="none")
        return (ce * w).sum() / w.sum()

    def normal_step(c):
        text = cache[c["key"]].strip()
        if not text:
            return None
        return ce_example(c["wav"], text)

    def quiet_step(rec):
        """Low-amplitude speech POSITIVE (user requirement: stay sensitive to
        quiet speech while robust to silence). Audio attenuated to the stored
        gain; target = the f32 teacher's transcript of the ATTENUATED audio.
        Trains exactly like a normal example -- full transcription."""
        wav = read_fixed_chunk(rec["audio_path"], rec["offset_s"],
                               rec["duration_s"], fe.sampling_rate)
        wav = np.clip(wav * rec["gain"], -1.0, 1.0).astype(np.float32)
        return ce_example(wav, rec["text"])

    # Recovery prefixes: the step-500 real-GGUF smoke showed that supervising
    # ONLY the first-token EOS is insufficient -- under real q4_K (+ the q4
    # encoder QAT never sees) the first decision can still flip, and the model
    # has no trained recovery, so it free-runs garbage. Teacher-forcing EOS
    # AFTER plausible spurious prefixes builds exactly that recovery: even if
    # a marker slips out, the very next decision stops.
    SIL_PREFIXES = ["", "[", "[0", "[0.", "[0.00", "[0.00]", "[0.00][S01]",
                    "[0.00][S01][0.06]", "I"]

    def silence_step(rec):
        wav = read_fixed_chunk(rec["audio_path"], rec["offset_s"],
                               rec["duration_s"], fe.sampling_rate)
        prefix = sil_rng.choice(SIL_PREFIXES)
        try:
            encd = proc(text=prompt_text + prefix, audio=[wav],
                        return_tensors="pt")
        except Exception:
            return None
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in encd.items()}
        out = model(**batch)
        # logits at the LAST position predict the next token after the
        # (possibly garbage-prefixed) assistant text. Target: EOS -- stop now,
        # regardless of what already slipped out. Heavily upweighted CE.
        logit = out.logits[0, -1:, :]
        target_ = torch.tensor([eos_id], device=dev)
        ce = F.cross_entropy(logit.float(), target_, reduction="mean")
        return ce * args.silence_weight

    order = list(range(len(pool)))
    ep_rng = random.Random(13)
    sil_rng = random.Random(17)
    step = micro = 0
    running = {"n": 0.0, "s": 0.0, "q": 0.0}
    counts = {"n": 0, "s": 0, "q": 0}
    t0 = time.time()
    while step < args.steps:
        ep_rng.shuffle(order)
        for idx in order:
            if step >= args.steps:
                break
            r = sil_rng.random()
            if r < args.silence_frac:
                kind, loss = "s", silence_step(sil_rng.choice(sil_pool))
            elif r < args.silence_frac + args.quiet_frac:
                kind, loss = "q", quiet_step(sil_rng.choice(quiet_pool))
            else:
                kind, loss = "n", normal_step(pool[idx])
            if loss is None:
                continue
            (loss / args.accum).backward()
            running[kind] += loss.item(); counts[kind] += 1
            micro += 1
            if micro % args.accum == 0:
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    msg = " ".join(
                        f"ce_{k}={running[k]/max(1,counts[k]):.4f}(n={counts[k]})"
                        for k in ("n", "q", "s"))
                    print(f"step {step:5d}: {msg} lr={lr_at(step):.2e} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    for k in running:
                        running[k] = 0.0; counts[k] = 0
                if step % args.save_every == 0 or step == args.steps:
                    save(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
