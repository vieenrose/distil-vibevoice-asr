#!/usr/bin/env python
"""Encoder distillation v2: feature regression + KL through the FROZEN decoder.

WHY THIS EXISTS (read scripts/60 and scripts/59 first)
------------------------------------------------------
scripts/60 distilled whisper-small to short_cos 0.985 / long_cos 0.970 against
the teacher's VQAdaptor output -- and still failed the gate badly (ASCEND MER
0.42 vs teacher 0.10; English 0.878 vs 0.156; DER 0.44 vs 0.23).

scripts/59 had measured, by perturbing the teacher's own features with ISOTROPIC
noise, that cosine 0.93 was lossless with no cliff even at 0.60. English in the
failed run sat at cosine 0.9322 -- above that "lossless" line -- with 7x the
damage. The prediction that isotropic noise would be the harsher, conservative
case was exactly BACKWARDS:

  * isotropic noise spreads error evenly over 1024 dims, so the decoder-relevant
    subspace absorbs only its proportional share and most of the error lands in
    directions the decoder ignores;
  * a student's residual CONCENTRATES in precisely the directions it failed to
    learn, which are the informative ones.

Cosine cannot tell those apart. So:

  1. THE OBJECTIVE. Plain MSE/cosine weights all 1024 dims equally, which is not
     what the decoder cares about. We add KL between the frozen decoder's logits
     when fed TEACHER features vs STUDENT features, on the assistant positions.
     That penalises error in exactly the directions that change decoder output.
     (This does NOT contradict the v7 "KL fights fake-quant" lesson -- that
     antagonism was KL vs fake-quant, and nothing is quantised here.)
  2. THE MONITOR. We evaluate real ASCEND MER -- zh, en AND mixed -- during
     training. Cosine is still logged, but it is diagnostic, never a gate.
     English is watched specifically because it collapsed silently last time.

Both forwards share ONE model object with the encoder/adaptor attributes swapped,
so the 0.6B decoder is never duplicated and is provably identical across the two.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(d_model=768, encoder_layers=12, encoder_attention_heads=12,
             encoder_ffn_dim=3072)


def _load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def student_tokens(enc, ada, feats, merge=4):
    h = enc(feats, return_dict=True).last_hidden_state
    B, T, D = h.shape
    Tt = (T // merge) * merge
    return ada(h[:, :Tt, :].reshape(B, Tt // merge, D * merge))


def save_model(out_dir, teacher_dir, teacher, t_enc, t_ada, enc, ada, d_model):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # teacher currently has the STUDENT modules swapped in; restore first so the
    # deepcopy carries the real decoder, then attach deep copies of the student.
    teacher.model.whisper_encoder, teacher.model.vq_adaptor = t_enc, t_ada
    model = copy.deepcopy(teacher).float()
    model.model.whisper_encoder = copy.deepcopy(enc).float()
    model.model.vq_adaptor = copy.deepcopy(ada).float()
    for k, v in SMALL.items():
        setattr(model.config.audio_config, k, v)
    model.config.adaptor_input_dim = d_model * 4
    model.to(torch.bfloat16).save_pretrained(out)
    for f in Path(teacher_dir).iterdir():
        if f.suffix in {".py", ".jinja"} or f.name.startswith(
                ("tokenizer", "processor", "preprocessor", "generation")):
            if not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    del model
    torch.cuda.empty_cache()
    print(f"saved -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/moss_ft_zhtw_v7")
    ap.add_argument("--init-from", default="models/moss_v8_encsmall",
                    help="feature-distilled starting point (scripts/60 output)")
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--max-audio-s", type=float, default=30.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--dev-per-bucket", type=int, default=5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="models/moss_v8b_encsmall_kl")
    args = ap.parse_args()

    import soundfile as sf
    import pyarrow.parquet as pq
    from opencc import OpenCC
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.eval.mer import mer
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    s55 = _load_script("s55", ROOT / "scripts" / "55_v61_marker_ft.py")

    dev = torch.device(args.device)
    tdir = str(ROOT / args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        tdir, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(tdir, trust_remote_code=True)
    tok = proc.tokenizer
    tgt_sr = proc.feature_extractor.sampling_rate

    t_enc, t_ada = model.model.whisper_encoder, model.model.vq_adaptor

    # ---- student: start from the feature-distilled checkpoint ---------------
    prev = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.init_from), trust_remote_code=True, dtype=torch.float32)
    enc = copy.deepcopy(prev.model.whisper_encoder).float().to(dev)
    ada = copy.deepcopy(prev.model.vq_adaptor).float().to(dev)
    del prev
    for p in list(enc.parameters()) + list(ada.parameters()):
        p.requires_grad_(True)
    d_model = SMALL["d_model"]
    print(f"student encoder {sum(p.numel() for p in enc.parameters())/1e6:.1f}M "
          f"init from {args.init_from}", flush=True)

    # ---- data ---------------------------------------------------------------
    rows = []
    with open(ROOT / args.ivod_manifest) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("segments") and Path(rec.get("audio_path", "")).exists():
                rows.append(rec)
    rng = random.Random(7)
    rng.shuffle(rows)
    print(f"ivod records: {len(rows)}", flush=True)

    # ---- dev set: real MER, all three buckets (English watched closely) ------
    cc = OpenCC("s2t")
    tbl = pq.read_table(ROOT / "data/raw/ascend/main/test-00000-of-00001.parquet")
    buckets = {"zh": [], "en": [], "mixed": []}
    for r in tbl.to_pylist():
        if r["language"] in buckets and 2.0 <= r["duration"] <= 15.0:
            buckets[r["language"]].append(r)
    drng = random.Random(1234)
    dev_set = []
    for k, v in buckets.items():
        drng.shuffle(v)
        for r in v[:args.dev_per_bucket]:
            w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            w = np.asarray(w if w.ndim == 1 else w.mean(1), np.float32)
            if sr != tgt_sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, tgt_sr)
                w = resample_poly(w, tgt_sr // g, sr // g).astype(np.float32)
            dev_set.append((k, w, r["transcription"]))
    print(f"dev set: {len(dev_set)} utts ({args.dev_per_bucket}/bucket)",
          flush=True)

    def dev_mer():
        model.model.whisper_encoder, model.model.vq_adaptor = enc, ada
        enc.eval(); ada.eval()
        per = {}
        for lang, w, ref in dev_set:
            messages = [{"role": "user", "content": [
                {"type": "audio", "audio": "x.wav"},
                {"type": "text", "text": DEFAULT_PROMPT}]}]
            txt = proc.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
            inp = proc(text=txt, audio=[w], return_tensors="pt").to(
                dev, torch.float32)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=160, do_sample=False)
            g = proc.decode(o[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True)
            hyp = "".join(s.text for s in parse_transcript_lenient(g)) or g
            per.setdefault(lang, []).append(
                min(mer(cc.convert(ref), cc.convert(hyp)), 1.0))
        enc.train(); ada.train()
        model.model.whisper_encoder, model.model.vq_adaptor = t_enc, t_ada
        out = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
        out["all"] = round(float(np.mean(
            [x for v in per.values() for x in v])), 4)
        return out

    params = list(enc.parameters()) + list(ada.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    m0 = dev_mer()
    print(f"step     0: dev MER {m0}", flush=True)
    hist = [{"step": 0, "mer": m0}]

    ri = 0
    for step in range(1, args.steps + 1):
        # window_sample returns (None, None) when it cannot fit >=2 segments
        # fully inside the window -- common for a 30 s window on sparse 45-min
        # recordings. Resample rather than burning the step.
        segs = None
        for _ in range(25):
            rec = rows[ri % len(rows)]; ri += 1
            off, segs = s55.window_sample(rec, args.max_audio_s, rng)
            if segs:
                break
        if not segs:
            continue
        try:
            wav, sr = sf.read(rec["audio_path"], start=int(off * 24000),
                              frames=int(args.max_audio_s * 24000))
        except Exception:
            continue
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1), np.float32)
        if len(wav) < sr:
            continue
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        assistant = s55.target_text(s55.densify_segments(segs))
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        prompt_text = proc.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        full_text = prompt_text + assistant + tok.eos_token
        try:
            encd = proc(text=full_text, audio=[wav], max_length=args.max_len,
                        return_tensors="pt")
            n_prompt = proc(text=prompt_text, audio=[wav],
                            max_length=args.max_len,
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

        # ---- reference logits from TEACHER features (frozen decoder) --------
        model.model.whisper_encoder, model.model.vq_adaptor = t_enc, t_ada
        with torch.no_grad():
            tl = model(**batch).logits[:, n_prompt - 1:-1, :]
            t_tok = t_ada(model.model.time_merge(
                t_enc(feats, return_dict=True).last_hidden_state))

        # ---- student logits through the SAME frozen decoder ----------------
        model.model.whisper_encoder, model.model.vq_adaptor = enc, ada
        sl = model(**batch).logits[:, n_prompt - 1:-1, :]
        s_tok = student_tokens(enc, ada, feats)
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

        if step % args.eval_every == 0:
            with torch.no_grad():
                cs = F.cosine_similarity(s_tok, t_tok, dim=-1).mean().item()
            m = dev_mer()
            print(f"step {step:5d}: kl={kl.item():.4f} feat={feat.item():.4f} "
                  f"cos={cs:.4f} | dev MER {m}", flush=True)
            hist.append({"step": step, "kl": kl.item(), "cos": cs, "mer": m})
        if step % args.save_every == 0 or step == args.steps:
            save_model(ROOT / args.out, tdir, model, t_enc, t_ada, enc, ada,
                       d_model)
            (ROOT / args.out / "kl_history.json").write_text(
                json.dumps(hist, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
