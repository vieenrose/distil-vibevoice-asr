#!/usr/bin/env python
"""v10b: LoRA compensation on the FROZEN decoder for the prune16 encoder.

Motivated by Kolluri et al. 2026 (arXiv 2603.27981, "On the Role of Encoder
Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR"): in the frozen-
encoder -> projector -> frozen-LLM architecture (which MOSS-TD is), pruning
damage is largely recoverable on the DECODING side -- LoRA r=16 on the LLM's
attention projections let a 2-layer-pruned encoder BEAT the unpruned baseline.
Their compensation is linguistic (substitution/deletion fixes via the LLM's
priors), which matches v10's measured error texture exactly (rare-word
homophone swaps, crosstalk fragments) -- and their weak spot (no encoder
retraining, so deep pruning collapses) is covered by our feature-distilled
prune16.

Design (choices that correct v9b's failure modes):
  * ENCODER + ADAPTOR FROZEN at v10 prune16 weights. Only LoRA (q/k/v/o of
    every decoder attention layer, r=16 a=32) trains -- ~few M params.
  * Objective: CE on TEACHER transcripts (the base model's own greedy output
    on each chunk, cached -- same self-labeling as v9b, no external labels),
    with STRUCTURAL TOKENS UPWEIGHTED 4x ('['/']'/digits/'.'/Sxx), the
    script-55 trick that historically protected diarization/markers.
  * Gradient ACCUMULATION 8 (v9b's per-sample KL at batch=1 spiked 0.25->18
    step to step; CE + accumulation is the stable recipe of scripts 55/63).
  * At save time the LoRA is MERGED into the decoder weights -> a plain full
    model dir, single-GGUF deployment unchanged.

Gate: scripts/87 probe per save, scripts/86 battery on the final -- same
teacher-equality bands as every encoder candidate.
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


AMI_WORDS = Path("/tmp/claude-1001/ami_ann/words")


def ami_gt_text(path, off, dur):
    """Ground-truth chunk target in MOSS marker format, from AMI word XML.

    v10b post-mortem: CE on teacher SELF-labels taught the LoRA the teacher's
    own English mistakes (en WER 0.18->0.26) while repairing zh (whose
    self-labels are near-clean). The paper's recipe uses ground truth; for
    AMI chunks we have it -- words with times per channel -- so English
    trains on TRUTH and only zh keeps self-labels."""
    import re as _re
    meeting = Path(path).stem
    words = []
    for ch in "ABCDE":
        f = AMI_WORDS / f"{meeting}.{ch}.words.xml"
        if not f.exists():
            continue
        for m in _re.finditer(
                r'<w[^>]*starttime="([\d.]+)"[^>]*endtime="([\d.]+)"'
                r'([^>]*)>([^<]+)</w>', f.read_text(errors="replace")):
            import html as _html
            s, e, attrs, txt = (float(m.group(1)), float(m.group(2)),
                                m.group(3), _html.unescape(m.group(4)))
            if s < off - 0.2 or s > off + dur - 0.05:
                continue
            words.append((s, e, ch, txt, 'punc="true"' in attrs))
    if not words:
        return ""
    words.sort(key=lambda w: (w[0], w[1]))
    turns = []  # (start, end, ch, text)
    for s, e, ch, txt, punc in words:
        if turns and turns[-1][2] == ch and s - turns[-1][1] <= 1.0:
            st, en, c, t = turns[-1]
            turns[-1] = (st, max(en, e), c, t + txt if punc else t + " " + txt)
        else:
            turns.append((s, max(s, e), ch, txt))
    spk = {}
    out = []
    for s, e, ch, t in turns:
        if ch not in spk:
            spk[ch] = f"S{len(spk) + 1:02d}"
        rs, re_ = max(0.0, s - off), max(0.0, min(dur, e - off))
        out.append(f"[{rs:.2f}][{spk[ch]}]{t.strip()}[{re_:.2f}]")
    return "".join(out)


def structural_ids(tok):
    ids = set()
    for s in list("[]0123456789.") + [f"S{i:02d}" for i in range(1, 12)]:
        for i in tok(s, add_special_tokens=False).input_ids:
            ids.add(i)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default="models/moss_v10_prune16_base")
    ap.add_argument("--teacher", default=None, help="base snapshot for self-labels")
    ap.add_argument("--manifest", default="/tmp/claude-1001/train_audio_manifest_b.jsonl")
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--short-ratio", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--struct-weight", type=float, default=4.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="models/moss_v10b_prune16_lora")
    ap.add_argument("--target-cache", default="/tmp/claude-1001/v9b_targets.jsonl")
    ap.add_argument("--ami-gt", action="store_true",
                    help="AMI chunks train on ground-truth word annotations "
                         "(MOSS marker format) instead of teacher self-labels")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    from peft import LoraConfig, get_peft_model

    import glob as _g
    base = args.teacher or _g.glob(
        "/home/luigi/.cache/huggingface/hub/"
        "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/")[0]
    dev = torch.device(args.device)
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tok = proc.tokenizer
    fe = proc.feature_extractor

    # Student: full v10 model (prune16 encoder + base decoder), all frozen.
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.init_from), trust_remote_code=True,
        dtype=torch.float32).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.train()

    # ---- pool (same seed/order as v9b => the target cache is reusable) -----
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
        pool.append({"wav": wav, "path": p, "off": off,
                     "key": f"{len(pool)}:{off:.2f}:{len(wav)}"})
    n_ami = sum(1 for c in pool if "ami_train" in c["path"])
    print(f"pool: {len(pool)} chunks ({n_ami} AMI)", flush=True)

    cache = {}
    cpath = Path(args.target_cache)
    if cpath.exists():
        for line in cpath.open():
            r = json.loads(line)
            cache[r["key"]] = r["text"]
    if args.ami_gt:
        n_gt = 0
        for c in pool:
            if "ami_train" in c["path"]:
                gt = ami_gt_text(c["path"], c["off"],
                                 len(c["wav"]) / fe.sampling_rate)
                cache[c["key"]] = gt  # overrides any self-label; "" skips chunk
                n_gt += 1
        print(f"AMI ground-truth targets: {n_gt}", flush=True)
    missing = [c for c in pool if c["key"] not in cache]
    if missing:
        print(f"self-labeling {len(missing)} chunks with the base teacher…",
              flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        ptxt = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        with cpath.open("a") as cf:
            for c in missing:
                encd = proc(text=ptxt, audio=[c["wav"]], return_tensors="pt")
                encd = {k: (v.to(dev) if torch.is_tensor(v) else v)
                        for k, v in encd.items()}
                n_prompt = encd["input_ids"].shape[1]
                with torch.no_grad():
                    out = teacher.generate(
                        **encd, do_sample=False,
                        max_new_tokens=int(12 * len(c["wav"]) / fe.sampling_rate) + 64,
                        pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
                cache[c["key"]] = text
                cf.write(json.dumps({"key": c["key"], "text": text},
                                    ensure_ascii=False) + "\n")
                cf.flush()
        del teacher
        torch.cuda.empty_cache()
    print("self-labels ready", flush=True)

    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": DEFAULT_PROMPT}]}]
    prompt_text = proc.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    sids = structural_ids(tok)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def save_merged(step):
        merged = model.merge_and_unload()  # returns the base model, LoRA folded in
        out = ROOT / args.out
        merged.save_pretrained(out)
        for f in Path(base).iterdir():
            if f.suffix in {".py", ".jinja"} or f.name.startswith(
                    ("tokenizer", "processor", "preprocessor", "generation")):
                if not (out / f.name).exists():
                    shutil.copy2(f, out / f.name)
        print(f"step {step}: merged+saved -> {out}", flush=True)
        # merge_and_unload mutates: re-wrap to continue training
        return get_peft_model(merged, lcfg)

    order = list(range(len(pool)))
    ep_rng = random.Random(13)
    step = 0
    micro = 0
    t0 = time.time()
    running = 0.0
    while step < args.steps:
        ep_rng.shuffle(order)
        for idx in order:
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

            out = model(**batch)
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
                if step % 50 == 0:
                    print(f"step {step:5d}: ce={running / (50 * args.accum):.4f} "
                          f"lr={lr_at(step):.2e} ({time.time() - t0:.0f}s)",
                          flush=True)
                    running = 0.0
                if step % args.save_every == 0 or step == args.steps:
                    model = save_merged(step)
                    model.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
