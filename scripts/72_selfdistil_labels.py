#!/usr/bin/env python
"""Self-distillation labels: BASE MOSS-TD transcribes the pool, its own output
becomes the training target.

Why self-distillation rather than a stronger teacher (whisper-large + pyannote,
which is what ivod_ft_v4.jsonl used):

  * Coverage is 100% by construction. v4 labelled only ~58% of each window's
    audio and the gaps were speech (gap RMS 0.396 vs labelled 0.441), so half of
    training taught "speech in, nothing out" -- the direct ancestor of the
    dense-speech marker collapse we spent today diagnosing.
  * Speaker tags come from the same forward pass as the text, so there is no
    pyannote-to-MOSS fusion step and no dominance filter dropping boundary
    segments (scripts/38's min_dominance 0.75 / min_coverage 0.7).
  * The target register is base's own, so the structural-token margin has no
    reason to move. Every FT in our lineage collapsed base's +4.90 margin to
    ~0.98, and below ~1 logit quantization noise flips marker decisions.

The risk this trades for: self-distillation cannot exceed the teacher, and it
reinforces base's own errors. That is acceptable for stage 2, whose goal is
zh-TW ORTHOGRAPHY with no regression -- not better recognition. Stage 3's
zh/en data is what is supposed to move accuracy.

QUALITY GATE. Every generated window is checked before it is written, because
the v4 corpus's defect was invisible until we measured it months later:
  coverage   labelled span / window duration      >= --min-coverage
  density    chars per covered second             within [--min-cps, --max-cps]
  structure  at least one [start][Sxx]...[end]    (a marker-less window is the
             exact collapse mode we are trying not to teach)
  loops      no 3-gram repeated > --max-rep times (base loops on dense speech)
Rejects are counted by reason and reported, so a pool that silently degrades
shows up as a rejection-rate spike rather than as a bad model three days later.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*(\[S\d+\])?\s*([^\[]*)\[(\d+(?:\.\d+)?)\]")


def parse_segments(s: str):
    """-> [(start, spk, text, end)]. Timestamps are window-relative seconds."""
    return [(float(a), spk or "", t.strip(), float(b))
            for a, spk, t, b in SEG_RE.findall(s)]


def has_loop(text: str, n: int = 3, max_rep: int = 4) -> bool:
    """CONSECUTIVE n-gram repetition, not total frequency.

    A total-frequency test rejected half of a smoke batch: over a 180 s window
    of parliamentary Chinese, ordinary phrases (委員會, 我們的) legitimately recur
    five or more times. The pathology we actually care about is the decoder
    latching -- the same n-gram back to back -- so measure runs, not counts.
    """
    toks = re.sub(r"\s+", "", text)
    if len(toks) < n * 2:
        return False
    run = 1
    for i in range(n, len(toks) - n + 1):
        if toks[i:i + n] == toks[i - n:i]:
            run += 1
            if run > max_rep:
                return True
        else:
            run = 1
    return False


def gen(model, proc, wav, dev, max_new_tokens):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/moss")
    ap.add_argument("--window", type=float, default=180.0)
    ap.add_argument("--hours", type=float, default=12.0,
                    help="audio hours to label before stopping")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--min-coverage", type=float, default=0.75)
    ap.add_argument("--min-cps", type=float, default=1.5)
    ap.add_argument("--max-cps", type=float, default=12.0)
    ap.add_argument("--max-rep", type=int, default=4)
    ap.add_argument("--zh-convert", default="s2tw",
                    help="opencc config; s2twp CORRUPTS proper nouns "
                         "(高端疫苗->高階疫苗), do not use it")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    import soundfile as sf
    from opencc import OpenCC
    from transformers import AutoModelForCausalLM, AutoProcessor

    cc = OpenCC(args.zh_convert)
    dev = torch.device(args.device)
    mdir = ROOT / args.model
    print(f"loading {mdir} (bf16) on {dev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(str(mdir), trust_remote_code=True)
    tsr = proc.feature_extractor.sampling_rate

    # ---- pool: zh parliamentary + en meetings, tagged so zh-TW conversion
    # applies only where it is meaningful.
    pool = [(p, "zh") for p in sorted((ROOT / "data/raw/ivod_ft").glob("*.wav"))]
    for p in sorted(Path("/tmp/claude-1001").glob("en_meet*.wav")):
        pool.append((p, "en"))
    if not pool:
        print("empty pool", file=sys.stderr)
        return 1
    print(f"pool: {sum(1 for _, l in pool if l == 'zh')} zh + "
          f"{sum(1 for _, l in pool if l == 'en')} en files", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fh = outp.open("w", encoding="utf-8")
    rej = Counter()
    kept = 0
    labelled_s = 0.0
    budget_s = args.hours * 3600
    t0 = time.time()

    for path, lang in pool:
        if labelled_s >= budget_s:
            break
        try:
            info = sf.info(str(path))
        except Exception as e:
            rej["unreadable"] += 1
            continue
        nwin = int(info.duration // args.window)
        for w in range(nwin):
            if labelled_s >= budget_s:
                break
            off = w * args.window
            try:
                wav, sr = sf.read(str(path), start=int(off * info.samplerate),
                                  frames=int(args.window * info.samplerate),
                                  dtype="float32")
            except Exception:
                rej["read_fail"] += 1
                continue
            wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)
            if sr != tsr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sr), int(tsr))
                wav = resample_poly(wav, tsr // g, sr // g).astype(np.float32)
            # silence gate: an all-silence window teaches nothing and base
            # hallucinates on it
            if float(np.sqrt((wav ** 2).mean())) < 1e-3:
                rej["silent"] += 1
                continue

            try:
                text = gen(model, proc, wav, dev, args.max_new_tokens)
            except Exception as e:
                rej[f"gen:{type(e).__name__}"] += 1
                continue

            segs = parse_segments(text)
            if not segs:
                rej["no_structure"] += 1
                continue
            covered = sum(max(0.0, b - a) for a, _, _, b in segs)
            cov = covered / args.window
            nchars = sum(len(t) for _, _, t, _ in segs)
            cps = nchars / max(1e-6, covered)
            if cov < args.min_coverage:
                rej["low_coverage"] += 1
                continue
            if not (args.min_cps <= cps <= args.max_cps):
                rej["density"] += 1
                continue
            if has_loop("".join(t for _, _, t, _ in segs), max_rep=args.max_rep):
                rej["loop"] += 1
                continue

            if lang == "zh":
                text = cc.convert(text)
            fh.write(json.dumps({
                "audio": str(path), "offset": round(off, 3),
                "duration": args.window, "lang": lang, "text": text,
                "coverage": round(cov, 4), "cps": round(cps, 2),
                "nsegs": len(segs),
                "nspk": len({s for _, s, _, _ in segs if s}),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1
            labelled_s += args.window
            if kept % 20 == 0:
                el = time.time() - t0
                print(f"  kept={kept} ({labelled_s/3600:.2f}h) "
                      f"rej={sum(rej.values())} {el/60:.1f}min "
                      f"rtf={labelled_s/max(1,el):.1f}x", flush=True)

    fh.close()
    print(f"\n=== {outp} ===")
    print(f"kept {kept} windows = {labelled_s/3600:.2f} h")
    print(f"rejected {sum(rej.values())}: {dict(rej)}")
    if kept:
        rows = [json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines()]
        cov = np.array([r["coverage"] for r in rows])
        cps = np.array([r["cps"] for r in rows])
        seg = np.array([r["nsegs"] for r in rows])
        spk = np.array([r["nspk"] for r in rows])
        print(f"coverage  median {np.median(cov):.3f}  p10 {np.percentile(cov,10):.3f}"
              f"   (v4 corpus was 0.58 / 0.24 -- that is the bar to beat)")
        print(f"chars/s   median {np.median(cps):.2f}")
        print(f"segs/win  median {np.median(seg):.1f}  (base does ~14 on dense speech)")
        print(f"speakers  median {np.median(spk):.1f}  multi-spk windows "
              f"{100*float((spk>1).mean()):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
