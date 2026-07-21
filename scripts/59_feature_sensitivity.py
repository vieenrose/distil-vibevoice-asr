#!/usr/bin/env python
"""How much audio-feature error can the MOSS decoder absorb? (cosine -> quality)

scripts/58 showed a whisper-small encoder can regress the teacher's VQAdaptor
output to cos ~0.93 in 1500 steps. That answered CAPACITY. It did NOT answer
SUFFICIENCY: nobody knows what cosine the decoder actually needs. Without that
number a long distillation run is a shot in the dark -- hitting 0.96 is wasted
effort if the threshold is 0.99, and equally wasted if it is 0.90.

This script measures the threshold directly, with NO training. It wraps the
teacher's own vq_adaptor so its output is degraded to a CONTROLLED per-frame
cosine, then runs the real eval battery at each level:

  SHORT audio -- ASCEND MER (same sample/seed/normalisation as scripts/45).
  LONG  audio -- a multi-minute window scored on the things that actually broke
                 in past regressions: marker count/cadence, speaker count, and
                 repetition-loop detection. Encoder damage is not expected to
                 hurt both halves equally: long audio crosses 30 s chunk
                 boundaries and must hold speaker identity across them, so it
                 is the harder half and gates on its own.

Degradation model: per frame, pred = c*t_hat + sqrt(1-c^2)*n_hat, with n_hat a
Gaussian direction orthogonalised against t, rescaled to ||t||. That is ISOTROPIC
error, whereas a distilled student's residual is STRUCTURED (it correlates with
the signal manifold and concentrates on hard frames). Isotropic noise is the
harsher of the two at equal cosine, so the threshold this yields is a
CONSERVATIVE bound -- a real student at cosine X should do no worse than this
curve says, and probably better. Read it as "safe target", not "exact target".
"""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class CosineDegrade(nn.Module):
    """Wraps VQAdaptor; degrades its output to a target per-frame cosine."""

    def __init__(self, inner: nn.Module, cos: float, seed: int = 0):
        super().__init__()
        self.inner = inner
        self.cos = float(cos)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def forward(self, x):
        t = self.inner(x)
        if self.cos >= 0.99999:
            return t
        n = torch.randn(t.shape, generator=self.gen, dtype=torch.float32).to(t.device)
        n = n.to(t.dtype)
        # remove the component along t, per frame
        tn = torch.nn.functional.normalize(t, dim=-1)
        n = n - (n * tn).sum(-1, keepdim=True) * tn
        nn_ = torch.nn.functional.normalize(n, dim=-1)
        c = self.cos
        out = c * tn + (1.0 - c * c) ** 0.5 * nn_
        return out * t.norm(dim=-1, keepdim=True)


def gen_text(model, proc, wav, dev, max_new_tokens=256):
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": DEFAULT_PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=text, audio=[wav], return_tensors="pt").to(dev, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)


def long_metrics(gen: str):
    """Marker/speaker/repetition health on a long window."""
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient
    segs = parse_transcript_lenient(gen)
    if not segs:
        return {"n_markers": 0, "n_speakers": 0, "median_seg_s": 0.0,
                "max_rep": 0, "dup_ratio": 0.0, "chars": len(gen),
                "monotone": True}
    spans = [s.end - s.start for s in segs]
    texts = [s.text.strip() for s in segs if s.text.strip()]
    cnt = Counter(texts)
    max_rep = max(cnt.values()) if cnt else 0
    dup = 1.0 - (len(cnt) / len(texts)) if texts else 0.0
    # char-level tight loop: most common 12-gram frequency
    joined = "".join(texts)
    grams = Counter(joined[i:i + 12] for i in range(max(0, len(joined) - 12)))
    top_gram = max(grams.values()) if grams else 0
    monotone = all(segs[i].start <= segs[i + 1].start for i in range(len(segs) - 1))
    return {"n_markers": len(segs),
            "n_speakers": len({s.speaker for s in segs}),
            "median_seg_s": round(float(np.median(spans)), 2),
            "max_rep": max_rep, "dup_ratio": round(dup, 3),
            "top12gram": top_gram, "chars": len(joined), "monotone": monotone}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v7")
    ap.add_argument("--cosines", nargs="+", type=float,
                    default=[1.0, 0.99, 0.97, 0.95, 0.93, 0.90, 0.85])
    ap.add_argument("--per-bucket", type=int, default=15)
    ap.add_argument("--long-wav", default="data/eval_synth/long_meeting.wav")
    ap.add_argument("--long-offset", type=float, default=300.0)
    ap.add_argument("--long-dur", type=float, default=180.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="data/feature_sensitivity.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from opencc import OpenCC
    from transformers import AutoModelForCausalLM, AutoProcessor
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.eval.mer import mer
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient
    sys.path.insert(0, str(ROOT / "scripts"))

    dev = torch.device(args.device)
    mdir = str(ROOT / args.model)
    model = AutoModelForCausalLM.from_pretrained(
        mdir, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(mdir, trust_remote_code=True)
    tgt_sr = proc.feature_extractor.sampling_rate
    cc = OpenCC("s2t")

    # ---- short set: ASCEND, same selection as scripts/45 --------------------
    tbl = pq.read_table(ROOT / "data/raw/ascend/main/test-00000-of-00001.parquet")
    rows = tbl.to_pylist()
    import random
    rng = random.Random(0)
    buckets = {"zh": [], "en": [], "mixed": []}
    for r in rows:
        if r["language"] in buckets and 2.0 <= r["duration"] <= 15.0:
            buckets[r["language"]].append(r)
    short = []
    for k, v in buckets.items():
        rng.shuffle(v)
        short += [(k, r) for r in v[:args.per_bucket]]
    print(f"short set: {len(short)} utts", flush=True)

    short_wavs = []
    for lang, r in short:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
        short_wavs.append((lang, wav, r["transcription"]))

    # ---- long window --------------------------------------------------------
    lp = ROOT / args.long_wav
    info = sf.info(str(lp))
    off = min(args.long_offset, max(0.0, info.frames / info.samplerate - args.long_dur))
    lwav, lsr = sf.read(str(lp), start=int(off * info.samplerate),
                        frames=int(args.long_dur * info.samplerate))
    lwav = np.asarray(lwav if lwav.ndim == 1 else lwav.mean(1), np.float32)
    if lsr != tgt_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(lsr, tgt_sr)
        lwav = resample_poly(lwav, tgt_sr // g, lsr // g).astype(np.float32)
    print(f"long window: {len(lwav)/tgt_sr:.0f}s from {lp.name} @ {off:.0f}s",
          flush=True)

    orig_adaptor = model.model.vq_adaptor
    results = {}
    for c in args.cosines:
        model.model.vq_adaptor = CosineDegrade(orig_adaptor, c, seed=0).to(dev)
        # verify the degradation actually lands where we asked
        per = {}
        for lang, wav, ref in short_wavs:
            gen = gen_text(model, proc, wav, dev)
            hyp = "".join(s.text for s in parse_transcript_lenient(gen)) or gen
            per.setdefault(lang, []).append(min(mer(cc.convert(ref),
                                                    cc.convert(hyp)), 1.0))
        short_res = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
        short_res["all"] = round(float(np.mean(
            [x for v in per.values() for x in v])), 4)

        lgen = gen_text(model, proc, lwav, dev, max_new_tokens=1024)
        lres = long_metrics(lgen)

        results[f"{c:.2f}"] = {"short": short_res, "long": lres}
        print(f"cos={c:.2f}  short_all={short_res['all']:.4f} "
              f"zh={short_res.get('zh',-1):.3f} en={short_res.get('en',-1):.3f} "
              f"mix={short_res.get('mixed',-1):.3f} | long: "
              f"markers={lres['n_markers']} spk={lres['n_speakers']} "
              f"med={lres['median_seg_s']}s rep={lres['max_rep']} "
              f"12gram={lres['top12gram']} chars={lres['chars']}", flush=True)
        model.model.vq_adaptor = orig_adaptor

    (ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"\n{'cos':>5s} {'MER_all':>8s} {'zh':>6s} {'en':>6s} {'mix':>6s} "
          f"{'mark':>5s} {'spk':>4s} {'rep':>4s} {'12gr':>5s}")
    for c, v in results.items():
        s, l = v["short"], v["long"]
        print(f"{c:>5s} {s['all']:8.4f} {s.get('zh',-1):6.3f} "
              f"{s.get('en',-1):6.3f} {s.get('mixed',-1):6.3f} "
              f"{l['n_markers']:5d} {l['n_speakers']:4d} {l['max_rep']:4d} "
              f"{l['top12gram']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
