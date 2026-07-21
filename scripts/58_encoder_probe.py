#!/usr/bin/env python
"""Encoder-capacity probe: can a SMALLER encoder reproduce medium's interface?

The Whisper-medium encoder is 55% of MOSS-TD's wall-clock (measured: 84 ms per
audio-second at 4 threads on x86; the Pi4/phone-class number is worse). It is
also the only part we have never pruned.

The audio path is
    whisper_encoder -> time_merge(4x) -> VQAdaptor -> 1024-d tokens -> LLM
and the LLM only ever sees the VQAdaptor OUTPUT. So the honest target for a
replacement encoder is not "transcribe well" (sequence-level CE, ~1 signal per
utterance, and it drags the decoder off its learned feature manifold) but
"reproduce the teacher's 1024-d audio tokens frame by frame" -- ~375 dense
regression targets per 30 s chunk, no transcripts required, decoder frozen and
literally unable to tell the difference if the match is tight. Whatever speaker
information medium encodes is IN the target, so diarization is preserved by
construction rather than by hope.

This script does NOT train a deliverable. It answers one question cheaply,
before we spend real GPU time: does the student have the CAPACITY to fit
medium's feature manifold? Read the held-out per-frame cosine.

Students:
  small    -- openai/whisper-small encoder (12L x 768, 0.30x compute -> 3.3x),
              fresh VQAdaptor (3072->1024) since the merged width changes.
  prune12  -- teacher's own encoder keeping every other layer (12L x 1024,
              0.50x -> 2.0x), VQAdaptor copied verbatim. The conservative
              control: same width, same adaptor, so a poor result here means
              the probe setup is wrong, not that the student is too small.

All Whisper sizes share the mel front-end and conv stride, so both students
consume the IDENTICAL input_features as the teacher and emit 1500 frames ->
the 12.5 tok/s cadence and timestamp granularity are untouched either way.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_wavs(manifest: Path, n: int, seed: int) -> list[str]:
    paths = []
    with open(manifest) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            p = rec.get("audio_path")
            if p and Path(p).exists():
                paths.append(p)
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def mel_batch(paths, feat_ext, rng, chunk_s=30.0):
    """Random chunk_s window from each path -> (B, 80, 3000) input_features."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    sr_t = feat_ext.sampling_rate
    feats = []
    for p in paths:
        try:
            info = sf.info(p)
            dur = info.frames / info.samplerate
            off = rng.uniform(0, max(0.0, dur - chunk_s))
            wav, sr = sf.read(p, start=int(off * info.samplerate),
                              frames=int(chunk_s * info.samplerate))
        except Exception:
            continue
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
        if wav.size < sr:
            continue
        if sr != sr_t:
            g = gcd(sr, sr_t)
            wav = resample_poly(wav, sr_t // g, sr // g).astype(np.float32)
        feats.append(feat_ext(wav, sampling_rate=sr_t,
                              return_tensors="pt").input_features[0])
    if not feats:
        return None
    return torch.stack(feats)


def build_student(kind, teacher, dev):
    """Returns (encoder, adaptor, tag). Both are trainable; teacher is frozen."""
    import copy
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    t_enc = teacher.model.whisper_encoder
    t_ada = teacher.model.vq_adaptor

    if kind == "prune12":
        # Every-other-layer init: the standard layer-drop that preserves
        # function far better than random or contiguous truncation.
        cfg = copy.deepcopy(t_enc.config)
        cfg.encoder_layers = 12
        enc = WhisperEncoder(cfg)
        sd = t_enc.state_dict()
        keep = list(range(0, 24, 2))
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("layers."):
                idx = int(k.split(".")[1])
                if idx in keep:
                    new_sd[k.replace(f"layers.{idx}.", f"layers.{keep.index(idx)}.", 1)] = v
            else:
                new_sd[k] = v
        missing, unexpected = enc.load_state_dict(new_sd, strict=False)
        print(f"[prune12] missing={len(missing)} unexpected={len(unexpected)}")
        ada = copy.deepcopy(t_ada)          # same width -> reuse verbatim
    elif kind == "small":
        w = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", dtype=torch.float32)
        enc = w.model.encoder
        del w
        # merged width 768*4=3072 -> fresh first Linear; reuse the rest of the
        # adaptor (second Linear + LayerNorm are shape-compatible at 1024).
        from importlib import import_module
        import sys
        sys.path.insert(0, str(ROOT / "models" / "moss_ft_zhtw_v7"))
        VQAdaptor = type(t_ada)
        ada = VQAdaptor(input_dim=768 * 4, hidden_size=1024,
                        norm_eps=1e-6)
        with torch.no_grad():
            ada.layers[2].load_state_dict(t_ada.layers[2].state_dict())
            ada.layers[3].load_state_dict(t_ada.layers[3].state_dict())
    else:
        raise ValueError(kind)

    return enc.to(dev).float(), ada.to(dev).float()


def student_tokens(enc, ada, feats, merge=4):
    h = enc(feats, return_dict=True).last_hidden_state       # (B,1500,D)
    B, T, D = h.shape
    Tt = (T // merge) * merge
    m = h[:, :Tt, :].reshape(B, Tt // merge, D * merge)
    return ada(m)                                            # (B,375,1024)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/moss_ft_zhtw_v7")
    ap.add_argument("--manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--students", nargs="+", default=["prune12", "small"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-wavs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="data/encoder_probe.json")
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

    wavs = load_wavs(ROOT / args.manifest, args.n_wavs, seed=0)
    print(f"probe wavs: {len(wavs)}", flush=True)
    if len(wavs) < 20:
        print("not enough audio; aborting")
        return 1
    n_hold = max(4, len(wavs) // 10)
    hold, train = wavs[:n_hold], wavs[n_hold:]

    # Fixed held-out batch so the cosine curve is comparable across steps.
    ev_rng = random.Random(1234)
    ev_feats = mel_batch(hold[:8], fe, ev_rng).to(dev)

    results = {}
    for kind in args.students:
        print(f"\n===== student: {kind} =====", flush=True)
        enc, ada = build_student(kind, teacher, dev)
        n_par = sum(p.numel() for p in enc.parameters())
        print(f"encoder params: {n_par/1e6:.1f}M "
              f"(teacher 307M)", flush=True)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(ada.parameters()),
                                lr=args.lr, weight_decay=0.01)
        rng = random.Random(7)
        curve = []

        def evaluate():
            enc.eval(); ada.eval()
            with torch.no_grad():
                tgt = teacher.model.vq_adaptor(
                    teacher.model.time_merge(
                        teacher.model.whisper_encoder(
                            ev_feats, return_dict=True).last_hidden_state))
                pred = student_tokens(enc, ada, ev_feats)
                cos = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                rel = ((pred - tgt).norm() / tgt.norm()).item()
            enc.train(); ada.train()
            return cos, rel

        c0, r0 = evaluate()
        print(f"step    0: cos={c0:.4f} rel_err={r0:.4f}", flush=True)
        curve.append({"step": 0, "cos": c0, "rel": r0})

        for step in range(1, args.steps + 1):
            batch = rng.sample(train, min(args.batch, len(train)))
            feats = mel_batch(batch, fe, rng)
            if feats is None:
                continue
            feats = feats.to(dev)
            with torch.no_grad():
                tgt = teacher.model.vq_adaptor(
                    teacher.model.time_merge(
                        teacher.model.whisper_encoder(
                            feats, return_dict=True).last_hidden_state))
            pred = student_tokens(enc, ada, feats)
            # cosine term drives direction, MSE term fixes scale.
            loss = F.mse_loss(pred, tgt) + \
                (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(ada.parameters()), 1.0)
            opt.step()

            if step % args.eval_every == 0:
                c, r = evaluate()
                print(f"step {step:4d}: loss={loss.item():.4f} "
                      f"cos={c:.4f} rel_err={r:.4f}", flush=True)
                curve.append({"step": step, "cos": c, "rel": r})

        results[kind] = {"enc_params_M": round(n_par / 1e6, 1), "curve": curve,
                         "final_cos": curve[-1]["cos"],
                         "final_rel": curve[-1]["rel"]}
        del enc, ada, opt
        torch.cuda.empty_cache()

    (ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"\n{'student':10s} {'encM':>7s} {'final_cos':>10s} {'rel_err':>9s}")
    for k, v in results.items():
        print(f"{k:10s} {v['enc_params_M']:7.1f} {v['final_cos']:10.4f} "
              f"{v['final_rel']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
