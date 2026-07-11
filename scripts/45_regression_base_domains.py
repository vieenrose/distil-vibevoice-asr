#!/usr/bin/env python
"""Base-domain regression check for MOSS fine-tunes.

FT on zh-TW synthetic meetings must not silently degrade what the BASE model
was good at. Scores base vs FT models on a fixed ASCEND sample (real
conversational Mandarin/English code-switching, the base model's home turf),
per language bucket (zh / en / mixed).

MER is computed in a script-normalized space (both hyp and ref -> Traditional
via OpenCC s2t) so Simplified-vs-Traditional output is NOT counted as error —
we're testing recognition, not script choice. Markers like [start][Sxx] are
stripped from hypotheses first.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pick_sample(rows, per_bucket, min_dur=2.0, max_dur=15.0, seed=0):
    import random
    rng = random.Random(seed)
    buckets = {"zh": [], "en": [], "mixed": []}
    for r in rows:
        if r["language"] in buckets and min_dur <= r["duration"] <= max_dur:
            buckets[r["language"]].append(r)
    out = []
    for k, v in buckets.items():
        rng.shuffle(v)
        out += [(k, r) for r in v[:per_bucket]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["models/moss", "models/moss_ft_zhtw_v2",
                             "models/moss_ft_zhtw_v3"])
    ap.add_argument("--per-bucket", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="data/regression_ascend.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from opencc import OpenCC
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.eval.mer import mer
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    cc = OpenCC("s2t")
    tbl = pq.read_table(ROOT / "data/raw/ascend/main/test-00000-of-00001.parquet")
    sample = pick_sample(tbl.to_pylist(), args.per_bucket)
    print(f"ASCEND sample: {len(sample)} utts "
          f"({args.per_bucket}/bucket zh/en/mixed)")

    dev = torch.device(args.device)
    results = {}
    for mpath in args.models:
        if not (ROOT / mpath).exists():
            print(f"{mpath}: missing, skipped")
            continue
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / mpath), trust_remote_code=True, dtype="auto"
        ).to(torch.bfloat16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(str(ROOT / mpath),
                                             trust_remote_code=True)
        tgt_sr = proc.feature_extractor.sampling_rate
        per_bucket_scores: dict[str, list[float]] = {}
        for lang, r in sample:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
            if sr != tgt_sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, tgt_sr)
                wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
            messages = [{"role": "user", "content": [
                {"type": "audio", "audio": "x.wav"},
                {"type": "text", "text": DEFAULT_PROMPT}]}]
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=text, audio=[wav], return_tensors="pt").to(
                dev, model.dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256,
                                     do_sample=False)
            gen = proc.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            hyp = "".join(s.text for s in parse_transcript_lenient(gen)) or gen
            score = mer(cc.convert(r["transcription"]), cc.convert(hyp))
            per_bucket_scores.setdefault(lang, []).append(min(score, 1.0))
        results[mpath] = {k: round(float(np.mean(v)), 4)
                          for k, v in per_bucket_scores.items()}
        results[mpath]["all"] = round(float(np.mean(
            [x for v in per_bucket_scores.values() for x in v])), 4)
        print(f"{mpath}: {results[mpath]}")
        del model
        torch.cuda.empty_cache()

    (ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"\n{'model':38s} {'zh':>7s} {'en':>7s} {'mixed':>7s} {'all':>7s}")
    for m, r in results.items():
        print(f"{m:38s} {r.get('zh', -1):7.3f} {r.get('en', -1):7.3f} "
              f"{r.get('mixed', -1):7.3f} {r.get('all', -1):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
