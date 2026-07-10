#!/usr/bin/env python
"""Evaluate meeting-title generation.

For each held-out meeting: prompt for a title, then score by keyword overlap
against the ground-truth (domain type + topic). Reports:
  - type_hit:  generated title contains the correct meeting-type word
  - topic_hit: generated title contains the true topic (or its head noun)
  - clean:     output is a single short line (not a transcript dump)
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from distil_vibevoice.data.dialogue_scripts import _DOMAIN_TOPICS, _DOMAIN_ZH

INSTRUCTION = "請用一句話為這段會議產生標題，格式為「會議類型：主題」。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_title")
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto").to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    from moss_transcribe_diarize.inference_utils import process_audio_info

    recs = [json.loads(l) for l in open(args.manifest)][args.skip:args.skip + args.n]
    type_hits = topic_hits = clean = 0
    for rec in recs:
        dom = rec["meta"].get("domain", "")
        dom_zh = _DOMAIN_ZH.get(dom, "會議")
        full = " ".join(s["text"] for s in rec["segments"])
        cands = [tp for tp in _DOMAIN_TOPICS.get(dom, []) if tp in full]
        topic = max(cands, key=len) if cands else ""

        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": rec["audio_path"]},
            {"type": "text", "text": INSTRUCTION}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios = process_audio_info(messages, sampling_rate=proc.feature_extractor.sampling_rate)
        inputs = proc(text=text, audio=audios, return_tensors="pt").to(dev, model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 repetition_penalty=1.2)
        gen = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # metrics
        t_hit = dom_zh[:2] in gen or dom_zh in gen  # meeting-type word
        # topic head: keep CJK/latin, compare token overlap
        tp_hit = False
        if topic:
            key = topic.split("的")[-1].split(" ")[-1][:6] or topic[:6]
            tp_hit = key in gen or any(w in gen for w in topic.split() if len(w) > 2)
        is_clean = len(gen) < 60 and "[" not in gen  # not a transcript dump
        type_hits += t_hit
        topic_hits += tp_hit
        clean += is_clean
        print(f"true: {dom_zh}：{topic[:24]}")
        print(f"gen : {gen[:60]!r}  [type={'Y' if t_hit else 'N'} topic={'Y' if tp_hit else 'N'} clean={'Y' if is_clean else 'N'}]")

    n = len(recs)
    print(f"\nTITLE over {n}: type-hit {type_hits}/{n} ({type_hits/n:.0%}), "
          f"topic-hit {topic_hits}/{n} ({topic_hits/n:.0%}), clean {clean}/{n} ({clean/n:.0%})")


if __name__ == "__main__":
    main()
