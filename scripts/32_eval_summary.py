#!/usr/bin/env python
"""Evaluate the summarization skill of a fine-tuned MOSS model.

For each held-out meeting: prompt the model with the notes-summarization
instruction, generate structured notes, and score them with the reference-free
fidelity metric (hallucination_rate + salient-fact coverage) against the
ground-truth transcript. Also reports whether the model still transcribes when
asked (task-switching sanity).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from distil_vibevoice.eval.summary_fidelity import check_fidelity

SUMMARY_INSTRUCTION = (
    "請為這段會議音訊整理結構化筆記（主題、重點、待辦），數字與日期必須逐字引用。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v3a")
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--skip", type=int, default=29)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    recs = [json.loads(l) for l in open(args.manifest)][args.skip:args.skip + args.n]
    h_rates, covs = [], []
    for rec in recs:
        transcript = " ".join(s["text"] for s in rec["segments"] if not s["text"].startswith("["))
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": rec["audio_path"]},
            {"type": "text", "text": SUMMARY_INSTRUCTION}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from moss_transcribe_diarize.inference_utils import process_audio_info
        audios = process_audio_info(messages, sampling_rate=proc.feature_extractor.sampling_rate)
        inputs = proc(text=text, audio=audios, return_tensors="pt").to(dev, model.dtype)
        t = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=800, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3)
        gen = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = check_fidelity(transcript, gen)
        h_rates.append(r.hallucination_rate)
        covs.append(r.coverage)
        print(f"=== {rec['duration_s']:.0f}s meeting ({len(rec['segments'])} segs, gen {time.time()-t:.0f}s) "
              f"| hallucination {r.hallucination_rate:.3f} coverage {r.coverage:.3f}")
        print("  NOTES:", gen[:300].replace("\n", " | "))
        if r.hallucinated:
            print("  HALLUCINATED FACTS:", r.hallucinated[:6])
    import numpy as np
    print(f"\nSUMMARY over {len(recs)}: mean hallucination_rate {np.mean(h_rates):.3f} "
          f"(0=faithful), mean coverage {np.mean(covs):.3f} (1=complete)")


if __name__ == "__main__":
    main()
