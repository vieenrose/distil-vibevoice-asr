#!/usr/bin/env python
"""ASCEND MER on the actual q4 GGUFs (deployment quant), via moss-td-test.

Mirrors scripts/45 exactly — same parquet, same pick_sample(seed=0), same
mer() in script-normalized (Traditional) space — but runs the ggml q4 engine
instead of the fp HF model, so the number reflects what ships.
"""
from __future__ import annotations
import argparse, io, subprocess, tempfile, os
from pathlib import Path
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BIN = "/home/luigi/RapidSpeech.cpp/build-x86/moss-td-test"


def pick_sample(rows, per_bucket, min_dur=2.0, max_dur=15.0, seed=0):
    import random
    rng = random.Random(seed)
    buckets: dict[str, list] = {}
    for r in rows:
        a = r["audio"]["bytes"]
        try:
            info = sf.info(io.BytesIO(a))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if not (min_dur <= dur <= max_dur):
            continue
        lang = r.get("language", r.get("lang", "zh")) or "zh"
        buckets.setdefault(lang, []).append(r)
    out = []
    for k, v in buckets.items():
        rng.shuffle(v)
        out += [(k, r) for r in v[:per_bucket]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ggufs", nargs="+", required=True)
    ap.add_argument("--per-bucket", type=int, default=25)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from opencc import OpenCC
    from distil_vibevoice.eval.mer import mer
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    cc = OpenCC("s2t")
    tbl = pq.read_table(ROOT / "data/raw/ascend/main/test-00000-of-00001.parquet")
    sample = pick_sample(tbl.to_pylist(), args.per_bucket)
    print(f"ASCEND sample: {len(sample)} utts ({args.per_bucket}/bucket)", flush=True)

    # decode all sample audio once to 16k wavs
    wavs = []
    for lang, r in sample:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        sf.write(p, wav, 16000)
        wavs.append((lang, p, r["transcription"]))

    for gg in args.ggufs:
        per = {}
        for lang, p, ref in wavs:
            try:
                out = subprocess.run([BIN, gg, p], capture_output=True, text=True,
                                     timeout=300).stdout
                line = [l for l in out.splitlines() if "Qwen3ASR: " in l]
                gen = line[-1].split("Qwen3ASR: ", 1)[1] if line else ""
            except Exception:
                gen = ""
            hyp = "".join(s.text for s in parse_transcript_lenient(gen)) or gen
            score = mer(cc.convert(ref), cc.convert(hyp))
            per.setdefault(lang, []).append(min(score, 1.0))
        res = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
        res["all"] = round(float(np.mean([x for v in per.values() for x in v])), 4)
        print(f"{os.path.basename(gg)}: {res}", flush=True)

    for _, p, _ in wavs:
        try: os.unlink(p)
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
