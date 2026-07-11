#!/usr/bin/env python
"""Simulate the EXACT in-browser pipeline in Python + measure q4 accuracy.

Pipeline (mirrors the JS implementation 1:1):
  mel (Whisper 80-bin, 30 s chunks padded to 3000 frames)
  -> encoder.onnx        (audio embeddings, keep (frames/2)/4 per chunk)
  -> embedding.onnx      (prompt token embeddings)
  -> JS-side splice      (audio embeddings into <|audio_pad|> slots)
  -> decoder.onnx loop   (dynamic KV: past -> present chaining, greedy)
  -> lenient parse + s2tw

Modes:
  --graphs models/moss_web      quantized set (encoder.int8/embedding.int8/decoder.q4)
  --graphs models/moss_onnx_v2  fp32 set (parity reference for the loop itself)

Accuracy: transcribes windows of the held-out REAL meetings and (if
--ref-model) the same windows with the bf16 HF model, reporting MER between
q4-ONNX output and bf16 output plus segment/tag statistics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

AUDIO_PAD = 151671
PREFIX = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
          151644, 872, 198, 151669]
SUFFIX = [151670, 198, 14880, 44063, 111268, 46670, 61443, 17714, 108704,
          3837, 73157, 104383, 58362, 23031, 71618, 26606, 20450, 111420,
          33108, 104283, 17340, 72640, 9909, 58, 50, 15, 16, 60, 5373, 58,
          50, 15, 17, 60, 5373, 58, 50, 15, 18, 60, 1940, 7552, 111749,
          3837, 110644, 17714, 110019, 105761, 43815, 90395, 18493, 37474,
          100072, 111066, 80565, 20450, 111420, 3837, 23031, 104542, 117932,
          75882, 37474, 105761, 101121, 1773, 151645, 198, 151644, 77091, 198]
EOS = 151645


def find_file(d: Path, stem: str) -> Path:
    for suf in (".q4.onnx", ".int8.onnx", ".onnx"):
        p = d / (stem + suf)
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}*.onnx in {d}")


class WebPipeline:
    def __init__(self, graphs_dir: Path, threads: int = 8):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        opts = dict(sess_options=so, providers=["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(str(find_file(graphs_dir, "encoder")), **opts)
        self.emb = ort.InferenceSession(str(find_file(graphs_dir, "embedding")), **opts)
        self.dec = ort.InferenceSession(str(find_file(graphs_dir, "decoder")), **opts)
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_ins = [i.name for i in self.dec.get_inputs()]
        self.dec_outs = [o.name for o in self.dec.get_outputs()]
        self.n_layers = sum(1 for n in self.dec_outs if "key" in n or n.startswith("present"))
        # feature extractor only for mel (JS reimplements this)
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(
            str(ROOT / "models/moss_ft_zhtw_v2"), trust_remote_code=True)

    def mel(self, wav: np.ndarray) -> np.ndarray:
        fe = self.proc.feature_extractor
        m = fe(wav, sampling_rate=16000, return_tensors="np",
               padding="max_length")["input_features"][0]  # [80, 3000]
        return m

    def encode_audio(self, wav: np.ndarray) -> np.ndarray:
        """30s chunking exactly like the C++/JS impl."""
        out = []
        n = len(wav)
        for off in range(0, n, 16000 * 30):
            piece = wav[off: off + 16000 * 30]
            n_frames = int(len(piece) / 160)
            m = self.mel(piece)[None]  # padded to 3000
            e = self.enc.run(None, {self.enc_in: m})[0][0]  # [750, 1024]
            keep = (n_frames // 2) // 4
            out.append(e[:keep])
        return np.concatenate(out, 0)

    def embed(self, ids: list[int]) -> np.ndarray:
        arr = np.asarray([ids], dtype=np.int64)
        return self.emb.run(None, {self.emb.get_inputs()[0].name: arr})[0][0]

    def generate(self, wav: np.ndarray, max_new: int = 1024):
        afeat = self.encode_audio(wav.astype(np.float32))
        ids = PREFIX + [AUDIO_PAD] * len(afeat) + SUFFIX
        embs = self.embed(ids)
        pad0 = len(PREFIX)
        embs[pad0: pad0 + len(afeat)] = afeat  # splice
        # prefill
        n_l = 28
        empty = np.zeros((1, 8, 0, 128), dtype=np.float32)
        feeds = {"inputs_embeds": embs[None].astype(np.float32),
                 "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        for i in range(n_l):
            feeds[f"past_k_{i}"] = empty
            feeds[f"past_v_{i}"] = empty
        toks = []
        t0 = time.time()
        for step in range(max_new):
            outs = self.dec.run(None, feeds)
            logits = outs[0][0, -1]
            tok = int(np.argmax(logits))
            if tok == EOS:
                break
            toks.append(tok)
            past = outs[1:]
            cur_len = past[0].shape[2]
            e = self.embed([tok])[None]
            feeds = {"inputs_embeds": e.astype(np.float32),
                     "attention_mask": np.ones((1, cur_len + 1), dtype=np.int64)}
            for i in range(n_l):
                feeds[f"past_k_{i}"] = past[2 * i]
                feeds[f"past_v_{i}"] = past[2 * i + 1]
        dt = time.time() - t0
        text = self.proc.tokenizer.decode(toks)
        return text, len(toks), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="models/moss_web")
    ap.add_argument("--wav", default="data/raw/ivod_eval/ivod_2024_15362.wav")
    ap.add_argument("--take-s", type=float, nargs="+", default=[30.0, 60.0])
    ap.add_argument("--ref-model", default="models/moss_ft_zhtw_v2",
                    help="bf16 HF model for accuracy reference ('' skips)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--s2tw", action="store_true", help="script-normalize hyp+ref before MER")
    args = ap.parse_args()

    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.eval.mer import mer as mer_fn
    from distil_vibevoice.runtime.lenient_parser import parse_transcript_lenient

    wav24, sr = sf.read(args.wav)
    wav24 = np.asarray(wav24 if wav24.ndim == 1 else wav24.mean(1), np.float32)
    g = gcd(sr, 16000)
    wav = resample_poly(wav24, 16000 // g, sr // g).astype(np.float32)

    pipe = WebPipeline(ROOT / args.graphs, threads=args.threads)

    ref_texts = {}
    if args.ref_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.ref_model, trust_remote_code=True, dtype="auto"
        ).to(torch.bfloat16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(args.ref_model, trust_remote_code=True)
        for t in args.take_s:
            piece = wav[: int(t * 16000)]
            messages = [{"role": "user", "content": [
                {"type": "audio", "audio": "x.wav"},
                {"type": "text", "text": DEFAULT_PROMPT}]}]
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=text, audio=[piece], return_tensors="pt").to(
                dev, model.dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1024,
                                     do_sample=False)
            ref_texts[t] = proc.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _s2tw = None
    if getattr(args, "s2tw", False):
        from opencc import OpenCC
        _s2tw = OpenCC("s2tw")

    for t in args.take_s:
        piece = wav[: int(t * 16000)]
        text, n_tok, dt = pipe.generate(piece)
        segs = parse_transcript_lenient(text)
        print(f"\n=== take {t:.0f}s | {args.graphs} ===")
        print(f"{n_tok} tokens in {dt:.1f}s ({n_tok/dt:.1f} tok/s, "
              f"RTF {dt/t:.2f}) | {len(segs)} segments")
        print("text:", text[:220])
        if t in ref_texts:
            ref_plain = "".join(s.text for s in parse_transcript_lenient(ref_texts[t]))
            hyp_plain = "".join(s.text for s in segs)
            if _s2tw is not None:
                ref_plain, hyp_plain = _s2tw.convert(ref_plain), _s2tw.convert(hyp_plain)
            mer_v = mer_fn(ref_plain, hyp_plain)
            print(f"vs bf16 FT: MER={mer_v:.3f}")
            print("bf16:", ref_texts[t][:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
