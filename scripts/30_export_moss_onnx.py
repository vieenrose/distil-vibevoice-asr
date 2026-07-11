#!/usr/bin/env python
"""Export MOSS-Transcribe-Diarize to ONNX for the sherpa-onnx port.

Three graphs:
  encoder.onnx   : mel (B,128,T_mel) -> audio embeddings (B, T_tok, hidden)
                   [WhisperEncoder -> 4x time_merge -> VQAdaptor, folded]
  embedding.onnx : input_ids (B,S) -> embeddings (B,S,hidden)
  decoder.onnx   : inputs_embeds (B,S,hidden) + past KV -> logits + present KV
                   [Qwen3-0.6B with KV cache, inputs_embeds entry point]

Parity: compares ONNX vs PyTorch outputs (encoder embeddings + one decode step).
CPU-only. Run: .venv/bin/python scripts/30_export_moss_onnx.py [--parity-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = None  # set from --out in main()


class EncoderWrap(torch.nn.Module):
    """Whisper encoder + time_merge + VQAdaptor as one graph."""

    def __init__(self, moss):
        super().__init__()
        self.enc = moss.model.whisper_encoder
        self.adaptor = moss.model.vq_adaptor
        self.merge = int(moss.config.audio_merge_size)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        h = self.enc(input_features).last_hidden_state          # (B, T, D)
        B, T, D = h.shape
        T_trim = (T // self.merge) * self.merge
        h = h[:, :T_trim, :].reshape(B, T_trim // self.merge, D * self.merge)
        return self.adaptor(h)                                   # (B, T', hidden)


class DecoderWrap(torch.nn.Module):
    """Qwen3 decoder step: inputs_embeds + flat past KV -> logits + present KV."""

    def __init__(self, moss):
        super().__init__()
        self.lm = moss.model.language_model
        self.lm_head = moss.lm_head
        self.n_layers = moss.config.text_config.num_hidden_layers
        self.last_logits = False

    def forward(self, inputs_embeds, attention_mask, *flat_past):
        from transformers.cache_utils import DynamicCache
        past = None
        if len(flat_past) and flat_past[0].shape[2] > 0:
            past = DynamicCache()
            for i in range(self.n_layers):
                past.update(flat_past[2 * i], flat_past[2 * i + 1], i)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      past_key_values=past, use_cache=True, return_dict=True)
        hidden = out.last_hidden_state
        if self.last_logits:
            hidden = hidden[:, -1:, :]
        logits = self.lm_head(hidden)
        pkv = out.past_key_values
        present = []
        for i in range(self.n_layers):
            layer = pkv.layers[i]
            present += [layer.keys, layer.values]
        return (logits, *present)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss")
    ap.add_argument("--out", default="models/moss_onnx")
    ap.add_argument("--last-logits", action="store_true",
                    help="decoder emits logits for the LAST position only "
                         "(saves a huge lm_head pass + a (S,vocab) tensor at "
                         "prefill; required for 32-bit wasm where full-prefill "
                         "logits are ~1.4 GB)")
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoProcessor
    torch.manual_seed(0)
    moss = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    cfg_t = moss.config.text_config
    n_layers, n_kv = cfg_t.num_hidden_layers, cfg_t.num_key_value_heads
    head_dim = getattr(cfg_t, "head_dim", cfg_t.hidden_size // cfg_t.num_attention_heads)
    hidden = cfg_t.hidden_size
    print(f"text: {n_layers}L hidden{hidden} kv{n_kv} hd{head_dim} | mel bins {moss.config.audio_config.num_mel_bins}")

    enc = EncoderWrap(moss).eval()
    dec = DecoderWrap(moss).eval()
    dec.last_logits = bool(args.last_logits)
    emb = moss.model.language_model.embed_tokens

    mel = torch.randn(1, moss.config.audio_config.num_mel_bins, 3000)  # 30s window
    if not args.parity_only:
        print("exporting encoder.onnx ...")
        torch.onnx.export(enc, (mel,), str(OUT / "encoder.onnx"),
                          input_names=["mel"], output_names=["audio_embeds"],
                          dynamic_axes={"mel": {0: "B", 2: "T_mel"}, "audio_embeds": {0: "B", 1: "T_tok"}},
                          opset_version=17, dynamo=False)

        print("exporting embedding.onnx ...")
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        torch.onnx.export(emb, (ids,), str(OUT / "embedding.onnx"),
                          input_names=["input_ids"], output_names=["embeds"],
                          dynamic_axes={"input_ids": {0: "B", 1: "S"}, "embeds": {0: "B", 1: "S"}},
                          opset_version=17, dynamo=False)

        print("exporting decoder.onnx ...")
        S, P = 4, 8  # trace WITH non-empty past so cache inputs survive tracing
        ie = torch.randn(1, S, hidden)
        am = torch.ones(1, P + S, dtype=torch.long)
        past = [torch.randn(1, n_kv, P, head_dim) for _ in range(2 * n_layers)]
        in_names = ["inputs_embeds", "attention_mask"] + [f"past_{t}_{i}" for i in range(n_layers) for t in ("k", "v")]
        out_names = ["logits"] + [f"present_{t}_{i}" for i in range(n_layers) for t in ("k", "v")]
        dyn = {"inputs_embeds": {0: "B", 1: "S"}, "attention_mask": {0: "B", 1: "S_total"},
               "logits": {0: "B"} if args.last_logits else {0: "B", 1: "S"}}
        for i in range(n_layers):
            for t in ("k", "v"):
                dyn[f"past_{t}_{i}"] = {0: "B", 2: "P"}
                dyn[f"present_{t}_{i}"] = {0: "B", 2: "P1"}
        torch.onnx.export(dec, (ie, am, *past), str(OUT / "decoder.onnx"),
                          input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                          opset_version=17, dynamo=False)
        # tokenizer assets for C++ BPE
        proc.tokenizer.save_pretrained(OUT / "tokenizer")
        print("exported. sizes:")
        for f in OUT.glob("*.onnx"):
            print(f"  {f.name}: {f.stat().st_size/1e6:.0f} MB")

    # ---------------- parity ----------------
    import onnxruntime as ort
    print("\nparity: encoder ...")
    ref = enc(mel).detach().numpy()
    s = ort.InferenceSession(str(OUT / "encoder.onnx"), providers=["CPUExecutionProvider"])
    got = s.run(None, {"mel": mel.numpy()})[0]
    e1 = float(np.abs(ref - got).max())
    print(f"  encoder max|Δ| = {e1:.2e} (shape {got.shape})")

    print("parity: decoder prefill + 1 step ...")
    S = 6
    ie = torch.randn(1, S, hidden)
    am = torch.ones(1, S, dtype=torch.long)
    past0 = [torch.zeros(1, n_kv, 0, head_dim) for _ in range(2 * n_layers)]
    ref_out = dec(ie, am, *past0)
    sd = ort.InferenceSession(str(OUT / "decoder.onnx"), providers=["CPUExecutionProvider"])
    feed = {"inputs_embeds": ie.numpy(), "attention_mask": am.numpy()}
    for i in range(n_layers):
        feed[f"past_k_{i}"] = past0[2 * i].numpy()
        feed[f"past_v_{i}"] = past0[2 * i + 1].numpy()
    got_out = sd.run(None, feed)
    e2 = float(np.abs(ref_out[0].detach().numpy() - got_out[0]).max())
    print(f"  prefill logits max|Δ| = {e2:.2e}")
    # one cached step
    ie2 = torch.randn(1, 1, hidden)
    am2 = torch.ones(1, S + 1, dtype=torch.long)
    ref2 = dec(ie2, am2, *[ref_out[1 + j] for j in range(2 * n_layers)])
    feed2 = {"inputs_embeds": ie2.numpy(), "attention_mask": am2.numpy()}
    for i in range(n_layers):
        feed2[f"past_k_{i}"] = got_out[1 + 2 * i]
        feed2[f"past_v_{i}"] = got_out[2 + 2 * i]
    got2 = sd.run(None, feed2)
    e3 = float(np.abs(ref2[0].detach().numpy() - got2[0]).max())
    print(f"  cached-step logits max|Δ| = {e3:.2e}")
    ok = e1 < 1e-3 and e2 < 1e-3 and e3 < 1e-3
    print(f"\nPARITY {'PASS' if ok else 'FAIL'} (thresholds 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
