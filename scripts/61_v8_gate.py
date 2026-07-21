#!/usr/bin/env python
"""Acceptance gate for the light-encoder model (v8) vs its teacher (v7).

The goal for v8 was: LIGHTER encoder, ~LOSSLESS on BOTH long and short audio.
Cosine to the teacher's features is a training signal, not evidence -- and the
feature-perturbation curves in scripts/59 could not settle diarization because
the windows they happened to use were single-speaker or synthetic-clean. This
script settles it against ground truth.

Three gates, deliberately covering the two regimes the goal names plus the
capability that must not regress:

  SHORT  -- ASCEND MER, same sample/seed/normalisation as scripts/45.
  DIAR   -- simulated multi-speaker meetings with EXACT speaker ground truth
            (built from distinct Common Voice zh-TW speakers), scored with the
            project's own der() and speaker_consistency(). This is the gate the
            perturbation proxies could not provide.
  LONG   -- a real far-field IVOD window: marker count/cadence, speaker count,
            and repetition-loop detection (the v6.1 failure mode).

Both models are run through the SAME code path on the SAME audio, so the
comparison is apples-to-apples; absolute numbers matter less than the delta.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


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


def resamp(wav, sr, tgt):
    from math import gcd
    from scipy.signal import resample_poly
    if sr == tgt:
        return wav
    g = gcd(sr, tgt)
    return resample_poly(wav, tgt // g, sr // g).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["models/moss_ft_zhtw_v7", "models/moss_v8_encsmall"])
    ap.add_argument("--per-bucket", type=int, default=25)
    ap.add_argument("--n-meetings", type=int, default=12)
    ap.add_argument("--long-wav", default="data/raw/ivod_ft/ivod_2024_15804.wav")
    ap.add_argument("--long-offset", type=float, default=600.0)
    ap.add_argument("--long-dur", type=float, default=180.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="data/v8_gate.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from opencc import OpenCC
    from transformers import AutoModelForCausalLM, AutoProcessor
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.eval.mer import mer
    from distil_vibevoice.eval.der import der
    from distil_vibevoice.eval.consistency import speaker_consistency
    from distil_vibevoice.data.manifest import Segment
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    cc = OpenCC("s2t")
    dev = torch.device(args.device)

    # ---------------- short set (ASCEND, scripts/45 selection) --------------
    import random
    tbl = pq.read_table(ROOT / "data/raw/ascend/main/test-00000-of-00001.parquet")
    buckets = {"zh": [], "en": [], "mixed": []}
    for r in tbl.to_pylist():
        if r["language"] in buckets and 2.0 <= r["duration"] <= 15.0:
            buckets[r["language"]].append(r)
    rng = random.Random(0)
    short = []
    for k, v in buckets.items():
        rng.shuffle(v)
        short += [(k, r) for r in v[:args.per_bucket]]

    # ---------------- diarization set (exact ground truth) ------------------
    meetings = []
    with open(ROOT / "data/manifests/simulated.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if Path(rec["audio_path"]).exists() and \
                    len({s["speaker"] for s in rec["segments"]}) >= 2:
                meetings.append(rec)
            if len(meetings) >= args.n_meetings:
                break
    print(f"short={len(short)} utts  meetings={len(meetings)}", flush=True)

    # ---------------- long real window --------------------------------------
    lp = ROOT / args.long_wav
    info = sf.info(str(lp))
    off = min(args.long_offset,
              max(0.0, info.frames / info.samplerate - args.long_dur))
    lwav_raw, lsr = sf.read(str(lp), start=int(off * info.samplerate),
                            frames=int(args.long_dur * info.samplerate))
    lwav_raw = np.asarray(lwav_raw if lwav_raw.ndim == 1 else lwav_raw.mean(1),
                          np.float32)

    results = {}
    for mpath in args.models:
        mdir = ROOT / mpath
        if not mdir.exists():
            print(f"{mpath}: missing, skipped")
            continue
        print(f"\n===== {mpath} =====", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(mdir), trust_remote_code=True,
            dtype="auto").to(torch.bfloat16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(str(mdir), trust_remote_code=True)
        tsr = proc.feature_extractor.sampling_rate
        enc_p = sum(p.numel() for p in model.model.whisper_encoder.parameters())
        tot_p = sum(p.numel() for p in model.parameters())

        # ---- short
        per = {}
        for lang, r in short:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
            wav = resamp(wav, sr, tsr)
            gen = gen_text(model, proc, wav, dev)
            hyp = "".join(s.text for s in parse_transcript_lenient(gen)) or gen
            per.setdefault(lang, []).append(
                min(mer(cc.convert(r["transcription"]), cc.convert(hyp)), 1.0))
        srt = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
        srt["all"] = round(float(np.mean(
            [x for v in per.values() for x in v])), 4)

        # ---- diarization vs exact ground truth
        ders, cons = [], []
        for rec in meetings:
            wav, sr = sf.read(rec["audio_path"])
            wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
            wav = resamp(wav, sr, tsr)
            gen = gen_text(model, proc, wav, dev, max_new_tokens=512)
            hyp = [Segment(start=s.start, end=s.end, speaker=s.speaker,
                           text=s.text) for s in parse_transcript_lenient(gen)]
            ref = [Segment(start=s["start"], end=s["end"],
                           speaker=s["speaker"], text=s.get("text", ""))
                   for s in rec["segments"]]
            if not hyp:
                ders.append(1.0); cons.append(0.0); continue
            ders.append(der(ref, hyp))
            cons.append(speaker_consistency(ref, hyp))
        diar = {"der": round(float(np.mean(ders)), 4),
                "consistency": round(float(np.mean(cons)), 4),
                "n": len(ders)}

        # ---- long real window
        lgen = gen_text(model, proc, resamp(lwav_raw, lsr, tsr), dev,
                        max_new_tokens=1024)
        segs = parse_transcript_lenient(lgen)
        texts = [s.text.strip() for s in segs if s.text.strip()]
        joined = "".join(texts)
        grams = Counter(joined[i:i + 12] for i in range(max(0, len(joined) - 12)))
        lng = {"markers": len(segs),
               "speakers": len({s.speaker for s in segs}),
               "median_seg_s": round(float(np.median(
                   [s.end - s.start for s in segs])), 2) if segs else 0.0,
               "max_rep": max(Counter(texts).values()) if texts else 0,
               "top12gram": max(grams.values()) if grams else 0,
               "chars": len(joined)}

        results[mpath] = {"enc_params_M": round(enc_p / 1e6, 1),
                          "total_params_M": round(tot_p / 1e6, 1),
                          "short": srt, "diar": diar, "long": lng}
        print(f"  short MER all={srt['all']} zh={srt.get('zh')} "
              f"en={srt.get('en')} mixed={srt.get('mixed')}", flush=True)
        print(f"  diar  DER={diar['der']} consistency={diar['consistency']} "
              f"(n={diar['n']})", flush=True)
        print(f"  long  markers={lng['markers']} spk={lng['speakers']} "
              f"med={lng['median_seg_s']}s rep={lng['max_rep']} "
              f"12gram={lng['top12gram']} chars={lng['chars']}", flush=True)
        del model
        torch.cuda.empty_cache()

    (ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"\n{'model':32s} {'encM':>6s} {'MER':>7s} {'DER':>7s} "
          f"{'cons':>6s} {'mark':>5s} {'rep':>4s}")
    for m, v in results.items():
        print(f"{Path(m).name:32s} {v['enc_params_M']:6.1f} "
              f"{v['short']['all']:7.4f} {v['diar']['der']:7.4f} "
              f"{v['diar']['consistency']:6.3f} {v['long']['markers']:5d} "
              f"{v['long']['max_rep']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
