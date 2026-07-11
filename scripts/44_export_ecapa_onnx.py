#!/usr/bin/env python
"""Export the speechbrain ECAPA-TDNN speaker embedder to ONNX for the browser.

Wrapper takes a raw 16 kHz waveform [1, T] and returns the L2-normalized
192-dim embedding [1, 192] (feature extraction included in-graph). Parity is
checked against EcapaEmbedder on real speech, and an int8 (MatMul-only)
quantized copy is produced for the web demo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def patch_stft_for_export(stft) -> None:
    """Replace torch.stft (complex -> not ONNX-exportable) with an equivalent
    conv1d DFT using the module's own geometry/window. Output layout matches
    speechbrain STFT.forward: [B, time, n_freq, 2]."""
    import math

    import torch.nn.functional as F

    n_fft = stft.n_fft
    hop = stft.hop_length
    window = stft.window
    if window.shape[0] < n_fft:  # torch.stft centers a short window
        lpad = (n_fft - window.shape[0]) // 2
        window = F.pad(window, (lpad, n_fft - window.shape[0] - lpad))
    n_freq = n_fft // 2 + 1
    k = torch.arange(n_freq).unsqueeze(1) * torch.arange(n_fft).unsqueeze(0)
    ang = 2.0 * math.pi * k.double() / n_fft
    cos_k = (torch.cos(ang).float() * window).unsqueeze(1)
    sin_k = (-torch.sin(ang).float() * window).unsqueeze(1)

    def fwd(x):  # [B, T]
        pad = n_fft // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode="constant")
        re = F.conv1d(xp, cos_k.to(x.dtype), stride=hop)
        im = F.conv1d(xp, sin_k.to(x.dtype), stride=hop)
        return torch.stack([re, im], dim=-1).transpose(1, 2)

    stft.forward = fwd


class EcapaWrapper(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.mods = classifier.mods
        patch_stft_for_export(self.mods.compute_features.compute_STFT)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # [1, T]
        feats = self.mods.compute_features(wav)
        feats = self.mods.mean_var_norm(
            feats, torch.ones(wav.shape[0], device=wav.device))
        emb = self.mods.embedding_model(feats)  # [1, 1, 192]
        emb = emb.squeeze(1)
        return torch.nn.functional.normalize(emb, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/moss_web/ecapa.onnx")
    args = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache/speechbrain/spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"})
    clf.eval()
    wrap = EcapaWrapper(clf).eval()

    dummy = torch.randn(1, 16000 * 3) * 0.1
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, (dummy,), str(out_path),
        input_names=["wav"], output_names=["embedding"],
        dynamic_axes={"wav": {1: "T"}}, opset_version=17, dynamo=False)
    print(f"exported {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")

    # ---- parity vs the Python embedder on real speech ----------------------
    import onnxruntime as ort
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.runtime.embeddings import EcapaEmbedder

    ref_emb = EcapaEmbedder(device="cpu")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    wav24, sr = sf.read(ROOT / "data/raw/ivod_eval/ivod_2024_15362.wav")
    wav24 = np.asarray(wav24 if wav24.ndim == 1 else wav24.mean(1), np.float32)
    g = gcd(sr, 16000)
    wav = resample_poly(wav24, 16000 // g, sr // g).astype(np.float32)

    worst = 1.0
    for s0, s1 in [(5, 9), (10, 16), (31, 40), (60, 65)]:
        seg = wav[16000 * s0: 16000 * s1]
        a = ref_emb.embed(seg, 16000)
        b = sess.run(None, {"wav": seg[None]})[0][0]
        cos = float(np.dot(a, b))
        worst = min(worst, cos)
        print(f"  seg {s0}-{s1}s cosine(onnx, speechbrain) = {cos:.5f}")
    print("PARITY", "PASS" if worst > 0.999 else "FAIL")

    # ---- int8 quantized copy for the web -----------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = out_path.with_suffix("").with_suffix("")  # strip .onnx
    q_path = out_path.parent / "ecapa.int8.onnx"
    quantize_dynamic(str(out_path), str(q_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    sess_q = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
    seg = wav[16000 * 10: 16000 * 16]
    a = ref_emb.embed(seg, 16000)
    b = sess_q.run(None, {"wav": seg[None]})[0][0]
    b = b / np.linalg.norm(b)
    print(f"ecapa.int8.onnx ({q_path.stat().st_size/1e6:.0f} MB) "
          f"cosine vs speechbrain = {float(np.dot(a, b)):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
