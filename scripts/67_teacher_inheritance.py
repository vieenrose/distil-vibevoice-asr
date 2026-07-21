#!/usr/bin/env python
"""Did we inherit the pseudo-label TEACHER's errors, or learn the audio?

Hypothesis 3 for the "too specialized" complaint: the IVOD targets are not
ground truth -- they are VibeVoice-ASR 7B's transcription, fused with pyannote
speakers (scripts/38). So the FT distilled ONE model's vocabulary, register and
MISTAKES. The reference itself shows the symptom ("郭志鈞 設長", "下午40前").

Style similarity alone proves nothing: a model that simply transcribes the audio
correctly will also resemble the teacher wherever the teacher was right. The
discriminating signal is agreement on the teacher's ERRORS.

Design:
  arbiter  = an INDEPENDENT ASR (whisper) on the same audio
  suspect errors = spans where TEACHER and ARBITER disagree
  then ask, on those spans only:
      does our FT side with the TEACHER (inheritance)
      or with the ARBITER (it heard the audio)?
  base MOSS is the control -- it never saw the teacher, so its teacher-agreement
  rate is the no-inheritance baseline.

  inheritance = P(FT agrees with teacher | teacher != arbiter)
              - P(base agrees with teacher | teacher != arbiter)

A large positive gap means we learned the teacher rather than the speech. Near
zero means the specialization is domain/vocabulary, not teacher register --
which points at hypotheses 2/5 instead and changes the fix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def norm(t: str) -> str:
    return re.sub(r"[，。、！？；：,.!?;:\s《》「」（）()]", "", t or "")


def sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--n-segments", type=int, default=40)
    ap.add_argument("--arbiter", default="openai/whisper-small")
    ap.add_argument("--models", nargs="+",
                    default=["/tmp/claude-1001/v71_f16.gguf",
                             "/tmp/claude-1001/base_q4.gguf"])
    ap.add_argument("--disagree-below", type=float, default=0.75,
                    help="teacher/arbiter similarity below this = suspect error")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    import subprocess, tempfile, os
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import pipeline

    rows = [json.loads(l) for l in open(ROOT / args.manifest)]
    rows = [r for r in rows if r.get("segments")
            and Path(r.get("audio_path", "")).exists()]
    # pick clean single-speaker segments of a usable length
    cand = []
    for r in rows:
        for s in r["segments"]:
            if 4.0 <= s["end"] - s["start"] <= 20.0 and len(s["text"]) >= 15:
                cand.append((r["audio_path"], s))
    import random
    random.Random(0).shuffle(cand)
    cand = cand[:args.n_segments]
    print(f"probing {len(cand)} teacher-labelled segments", flush=True)

    asr = pipeline("automatic-speech-recognition", model=args.arbiter,
                   device=args.device)

    BIN = "/home/luigi/RapidSpeech.cpp/build-x86/moss-td-test"
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient
    sys.path.insert(0, str(ROOT / "src"))

    rec_teacher, rec_arb, rec_models = [], [], {m: [] for m in args.models}
    for path, seg in cand:
        try:
            info = sf.info(path)
            wav, sr = sf.read(path, start=int(seg["start"] * info.samplerate),
                              frames=int((seg["end"] - seg["start"]) * info.samplerate))
        except Exception:
            continue
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
        if sr != 16000:
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        if wav.size < 16000:
            continue
        try:
            arb = asr(wav.copy(), generate_kwargs={"language": "chinese"})["text"]
        except Exception:
            continue
        fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        sf.write(p, wav, 16000)
        outs = {}
        for m in args.models:
            try:
                o = subprocess.run([BIN, m, p], capture_output=True, text=True,
                                   timeout=300).stdout
                line = [l for l in o.splitlines() if "Qwen3ASR: " in l]
                gen = line[-1].split("Qwen3ASR: ", 1)[1] if line else ""
            except Exception:
                gen = ""
            outs[m] = "".join(s.text for s in parse_transcript_lenient(gen)) or gen
        os.unlink(p)
        rec_teacher.append(seg["text"]); rec_arb.append(arb)
        for m in args.models:
            rec_models[m].append(outs[m])

    n = len(rec_teacher)
    ta = [sim(t, a) for t, a in zip(rec_teacher, rec_arb)]
    suspect = [i for i in range(n) if ta[i] < args.disagree_below]
    print(f"\nteacher vs arbiter: median sim {np.median(ta):.3f}; "
          f"{len(suspect)}/{n} segments flagged as suspect teacher errors\n")
    print(f"{'model':34s} {'agree w/ TEACHER':>18s} {'agree w/ ARBITER':>18s}  (on suspect spans)")
    for m in args.models:
        hyp = rec_models[m]
        if not suspect:
            print(f"{Path(m).name:34s} (no suspect spans)"); continue
        at = np.mean([sim(hyp[i], rec_teacher[i]) for i in suspect])
        aa = np.mean([sim(hyp[i], rec_arb[i]) for i in suspect])
        print(f"{Path(m).name:34s} {at:18.3f} {aa:18.3f}")
    print("\nInheritance = (FT teacher-agreement) - (base teacher-agreement) on suspect spans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
