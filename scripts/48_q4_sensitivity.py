#!/usr/bin/env python
"""q4 sensitivity probe (mixed-precision search / 'NAS' for quantization).

Measures how much each decoder Linear-group hurts when fake-quantized to int4,
so the export can keep the most sensitive ones at int8 (mixed precision) and
q4 the rest. Sensitivity = increase in held-out CE loss when ONLY that group
is fake-quantized (everything else full precision).

Groups: per decoder layer (all 7 proj matmuls together) + lm_head separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v4")
    ap.add_argument("--manifest", default="data/pseudo/tts_all.jsonl")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--out", default="data/q4_sensitivity.json")
    args = ap.parse_args()

    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.quant.fakequant import fake_quant_int4

    dev = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.model), trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(str(ROOT / args.model),
                                         trust_remote_code=True)
    tgt_sr = proc.feature_extractor.sampling_rate

    # build a few fixed eval batches
    rows = [json.loads(l) for l in open(ROOT / args.manifest)][:args.n_batches]
    batches = []
    for rec in rows:
        wav, sr = sf.read(rec["audio_path"])
        wav = np.asarray(wav if wav.ndim == 1 else wav.mean(1), np.float32)[:120 * sr]
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)
        segs = [s for s in rec["segments"] if s["start"] < 120]
        if not segs:
            continue
        tgt = "".join(f"[{s['start']:.2f}][S{int(s['speaker'])+1:02d}]{s['text']}"
                      f"[{s['end']:.2f}]" for s in segs)
        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"}, {"type": "text", "text": DEFAULT_PROMPT}]}]
        ptext = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=ptext + tgt + proc.tokenizer.eos_token, audio=[wav],
                   max_length=4096, return_tensors="pt")
        npf = proc(text=ptext, audio=[wav], max_length=4096,
                   return_tensors="pt")["input_ids"].shape[1]
        batches.append((enc, npf))

    def eval_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for enc, npf in batches:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
                labels = b["input_ids"].clone()
                labels[:, :npf] = -100
                b["labels"] = labels
                tot += float(model(**b).loss)
                n += 1
        return tot / max(1, n)

    import torch.nn as nn
    base = eval_loss()
    print(f"baseline CE (fp) = {base:.4f}", flush=True)

    # collect target linears grouped by decoder layer index + lm_head
    groups: dict[str, list] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(
                t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj", "lm_head")):
            if "lm_head" in name:
                key = "lm_head"
            else:
                # .../layers.<i>/...
                import re
                m = re.search(r"layers\.(\d+)\.", name)
                key = f"layer{int(m.group(1)):02d}" if m else "other"
            groups.setdefault(key, []).append((name, mod))

    sens = {}
    for key, mods in sorted(groups.items()):
        orig = {}
        for name, mod in mods:
            orig[name] = mod.weight.data.clone()
            mod.weight.data = fake_quant_int4(mod.weight.data.float()).to(mod.weight.dtype)
        loss = eval_loss()
        for name, mod in mods:
            mod.weight.data = orig[name]
        sens[key] = round(loss - base, 5)
        print(f"  {key:9s} q4-only CE +{sens[key]:.5f}", flush=True)

    ranked = sorted(sens.items(), key=lambda kv: -kv[1])
    (ROOT / args.out).write_text(json.dumps(
        {"baseline": base, "sensitivity": sens, "ranked": ranked}, indent=1))
    print("\nmost q4-sensitive (keep int8):", [k for k, _ in ranked[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
