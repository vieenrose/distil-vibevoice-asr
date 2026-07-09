#!/usr/bin/env python
"""Evaluate MOSS-Transcribe-Diarize on our zh-TW meetings (audio + labels).

Runs MOSS transcription, compares vs reference segments with our MER/DER metrics.
Set CUDA_VISIBLE_DEVICES to the allowed GPU.
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
from distil_vibevoice.data.manifest import Segment
from distil_vibevoice.eval.mer import mer
from distil_vibevoice.eval.der import der


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained("models/moss", trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained("models/moss", trust_remote_code=True)
    recs = [json.loads(l) for l in open(args.manifest)][:args.n]
    mers, ders = [], []
    for rec in recs:
        ref = [Segment(s["start"], s["end"], s["speaker"], s["text"]) for s in rec["segments"]]
        t = time.time()
        out = generate_transcription(model, proc, build_transcription_messages(rec["audio_path"]),
                                     max_new_tokens=1500, do_sample=False, device=dev, dtype=torch.bfloat16)
        gen = time.time() - t
        try:
            hyp_raw = parse_transcript(out["text"])
            hyp = [Segment(getattr(s, "start", 0.0) or 0.0, getattr(s, "end", 0.0) or 0.0,
                           str(getattr(s, "speaker", "0")), getattr(s, "text", "")) for s in hyp_raw]
        except Exception as e:
            hyp = []
            print("  parse failed:", e)
        ref_txt = " ".join(s.text for s in ref)
        hyp_txt = " ".join(s.text for s in hyp)
        m = mer(ref_txt, hyp_txt) if hyp else 1.0
        try:
            d = der(ref, hyp) if hyp else 1.0
        except Exception:
            d = None
        mers.append(m); ders.append(d if d is not None else 1.0)
        print(f"=== {rec['duration_s']:.0f}s, {len(ref)} ref segs / {len(hyp)} hyp segs (gen {gen:.0f}s) | MER {m:.3f} DER {d if d is None else round(d,3)}")
        print("  REF:", ref_txt[:70])
        print("  HYP:", hyp_txt[:70])
    import numpy as np
    print(f"\nMOSS on {len(recs)} zh-TW meetings: mean MER {np.mean(mers):.3f}, mean DER {np.mean(ders):.3f}")


if __name__ == "__main__":
    main()
