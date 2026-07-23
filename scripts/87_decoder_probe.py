#!/usr/bin/env python3
"""Fast decoder-aware probe for encoder candidates. The anti-cosine.

One teacher-forced forward pass per clip (~seconds) instead of full
transcription + battery (~20 min). Everything is measured at the FROZEN
decoder's output, which is the only place encoder error matters:

  kl        mean KL(teacher logits || candidate logits) per forced token
  agree     % positions where candidate argmax == the forced (teacher) token
  s_margin  mean logit margin (top1 - top2) at STRUCTURAL positions
            ('[', ']', digits, Sxx) -- the metric whose collapse (4.90 -> 0.98)
            predicted every historical failure of this model family
  per-clip  zh / en separated: v8's English collapse was invisible in
            aggregate cosine and instant in per-domain decoder behaviour

Calibration contract: before trusting this probe, it must (a) score the
teacher itself perfectly (KL 0, agree 100%), and (b) FLAG models/
moss_v8_encsmall -- the preserved known-bad encoder that cosine 0.98 waved
through while English MER was 0.878. If it can't separate those two, it is
no better than cosine and must not be used as a gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")

CLIPS = [
    ("zh", "/tmp/claude-1001/golden/golden_zh_5min.wav",
     "/tmp/claude-1001/golden/ref_zh_f32.txt"),
    ("en", "/tmp/claude-1001/golden/golden_en_5min.wav",
     "/tmp/claude-1001/golden/ref_en_f32.txt"),
]


def load_model(path, device, encoder_from=None):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, dtype=torch.float32).to(device).eval()
    if encoder_from is not None:
        # Graft the encoder (+adaptor). If the candidate's encoder has a
        # different width (e.g. the 768-d v8 artifact), grafting is impossible
        # -- fall back to loading the candidate as a COMPLETE model. Note the
        # caveat: a complete legacy artifact carries its own decoder too, so
        # the probe then measures encoder+decoder jointly; fine for the
        # known-bad calibration, but v9 candidates are saved with the frozen
        # base decoder so the graft path is what runs for them.
        src = AutoModelForCausalLM.from_pretrained(
            str(encoder_from), trust_remote_code=True, dtype=torch.float32)
        try:
            m.model.whisper_encoder.load_state_dict(src.model.whisper_encoder.state_dict())
            if hasattr(src.model, "vq_adaptor"):
                m.model.vq_adaptor.load_state_dict(src.model.vq_adaptor.state_dict())
            del src
        except RuntimeError:
            del m
            m = src.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def forced_logits(model, proc, wav, ref_text, device):
    """Logits over the forced continuation (the byte-validated reference)."""
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages, prepare_inputs)
    messages = build_transcription_messages(wav)
    inputs = prepare_inputs(proc, messages, device=device)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    tok = proc.tokenizer
    forced = tok(ref_text, return_tensors="pt").input_ids.to(device)
    ids = torch.cat([inputs["input_ids"], forced], dim=1)
    fwd = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    out = model(input_ids=ids, **fwd)
    S = inputs["input_ids"].shape[1]
    # logits predicting forced[t] live at position S-1+t
    lg = out.logits[0, S - 1: S - 1 + forced.shape[1], :]
    return lg.float(), forced[0]


def structural_ids(tok):
    ids = set()
    for s in list("[]0123456789.") + [f"S{i:02d}" for i in range(1, 12)]:
        for i in tok(s, add_special_tokens=False).input_ids:
            ids.add(i)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", help="HF dir with the candidate ENCODER (grafted onto the base)")
    ap.add_argument("--base", default=None, help="teacher dir (default: HF cache snapshot)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import glob as _g
    base = args.base or _g.glob("/home/luigi/.cache/huggingface/hub/"
                                "models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/*/")[0]
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    dev = torch.device(args.device)

    teacher = load_model(base, dev)
    is_self = str(Path(args.candidate).resolve()) == str(Path(base).resolve())
    cand = teacher if is_self else load_model(base, dev, encoder_from=args.candidate)

    sids = structural_ids(proc.tokenizer)
    report = {"tag": args.tag or Path(args.candidate).name, "ts": time.strftime("%F %T")}
    t0 = time.time()
    for lang, wav, ref in CLIPS:
        ref_text = Path(ref).read_text().rstrip("\n")
        lg_t, forced = forced_logits(teacher, proc, wav, ref_text, dev)
        lg_c = lg_t if is_self else forced_logits(cand, proc, wav, ref_text, dev)[0]

        p_t = torch.log_softmax(lg_t, -1)
        p_c = torch.log_softmax(lg_c, -1)
        kl = torch.sum(p_t.exp() * (p_t - p_c), -1)          # [T]
        agree = (lg_c.argmax(-1) == forced).float()

        smask = torch.tensor([int(t.item()) in sids for t in forced], device=dev)
        top2 = lg_c.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        report[f"{lang}_kl"] = round(kl.mean().item(), 4)
        report[f"{lang}_agree"] = round(100 * agree.mean().item(), 2)
        if smask.any():
            report[f"{lang}_s_margin"] = round(margin[smask.bool()].mean().item(), 3)
            report[f"{lang}_s_agree"] = round(100 * agree[smask.bool()].mean().item(), 2)
    report["probe_s"] = round(time.time() - t0, 1)
    line = json.dumps(report)
    print(line)
    with open("/tmp/claude-1001/decoder_probe_log.jsonl", "a") as f:
        f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
