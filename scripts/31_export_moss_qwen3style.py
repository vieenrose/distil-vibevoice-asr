#!/usr/bin/env python
"""Export the MOSS decoder to the sherpa-onnx Qwen3-ASR graph interface.

Interface (introspected from csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8):
  inputs : input_ids[B,S] i64, audio_features[B,A,1024] f32,
           attention_mask[B,S] i64 (accepted, unused: mask derives from
           cache_position), cache_position[S] i64,
           cache_key_i/cache_value_i [B, max_total_len, 8, 128]  (28 layers)
  outputs: logits[B,S,vocab], key_delta_i/value_delta_i [B,S,8,128]

Semantics: audio embeddings are scattered INSIDE the graph at positions where
input_ids == audio_token_id (151671). Attention runs over the fixed-size cache
with current K/V written at cache_position; visibility mask = column_global_pos
<= query_global_pos (contiguous cache). The C++ side (ApplyKvDeltaInplace)
writes the returned deltas into its cache copy at the same positions.

Parity: prefill + one cached decode step compared against the stock HF forward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = None  # set from --out in main()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class FixedCacheMossDecoder(nn.Module):
    """Qwen3 decoder re-expressed over a fixed-size KV cache (sherpa layout)."""

    def __init__(self, moss, max_total_len: int):
        super().__init__()
        self.lm = moss.model.language_model
        self.lm_head = moss.lm_head
        cfg = moss.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.n_q = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.audio_token_id = int(moss.config.audio_token_id)
        self.max_total_len = max_total_len
        self.scale = self.hd ** -0.5

    def forward(self, input_ids, audio_features, attention_mask, cache_position,
                *flat_cache):
        B, S = input_ids.shape
        h = self.lm.embed_tokens(input_ids)                      # [B,S,H]
        # scatter audio features at placeholder positions (masked_scatter)
        audio_mask = (input_ids == self.audio_token_id)          # [B,S]
        h = h.masked_scatter(audio_mask.unsqueeze(-1),
                             audio_features.to(h.dtype).reshape(-1, h.shape[-1]))

        pos_ids = cache_position.unsqueeze(0)                     # [1,S]
        cos, sin = self.lm.rotary_emb(h, pos_ids)                 # [1,S,hd]
        cos = cos.unsqueeze(1)                                    # [1,1,S,hd]
        sin = sin.unsqueeze(1)

        # visibility: column global position j visible to query with global
        # position p when j <= p (cache is contiguously filled)
        col = torch.arange(self.max_total_len, device=input_ids.device)
        vis = col.unsqueeze(0) <= cache_position.unsqueeze(1)     # [S,L]
        neg = torch.finfo(h.dtype).min

        deltas = []
        for i, layer in enumerate(self.lm.layers):
            k_cache = flat_cache[2 * i]                           # [B,L,kv,hd]
            v_cache = flat_cache[2 * i + 1]
            resid = h
            x = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(x).view(B, S, self.n_q, self.hd))
            k = attn.k_norm(attn.k_proj(x).view(B, S, self.n_kv, self.hd))
            v = attn.v_proj(x).view(B, S, self.n_kv, self.hd)
            # RoPE in [B,heads,S,hd]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k_delta = k.transpose(1, 2)                           # [B,S,kv,hd]
            v_delta = v
            deltas += [k_delta, v_delta]
            # write current K/V into the fixed cache at cache_position
            k_full = k_cache.index_copy(1, cache_position, k_delta)  # [B,L,kv,hd]
            v_full = v_cache.index_copy(1, cache_position, v_delta)
            # attention (GQA): expand kv heads to q heads
            kf = k_full.permute(0, 2, 1, 3)                       # [B,kv,L,hd]
            vf = v_full.permute(0, 2, 1, 3)
            rep = self.n_q // self.n_kv
            kf = kf.repeat_interleave(rep, dim=1)                 # [B,nq,L,hd]
            vf = vf.repeat_interleave(rep, dim=1)
            scores = torch.matmul(q, kf.transpose(-1, -2)) * self.scale  # [B,nq,S,L]
            scores = scores.masked_fill(~vis.unsqueeze(0).unsqueeze(0), neg)
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vf)                         # [B,nq,S,hd]
            ctx = ctx.transpose(1, 2).reshape(B, S, self.n_q * self.hd)
            h = resid + attn.o_proj(ctx)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        h = self.lm.norm(h)
        logits = self.lm_head(h)
        return (logits, *deltas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss")
    ap.add_argument("--out", default="models/moss_onnx_sherpa")
    ap.add_argument("--max-total-len", type=int, default=8192)
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    moss = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                dtype=torch.float32).eval()
    dec = FixedCacheMossDecoder(moss, args.max_total_len).eval()
    L = args.max_total_len
    n_layers, n_kv, hd = dec.n_layers, dec.n_kv, dec.hd
    hidden = moss.config.text_config.hidden_size

    # ------------------------- parity vs stock HF -------------------------
    S, A = 10, 4
    ids = torch.randint(1000, 2000, (1, S))
    ids[0, 2:2 + A] = dec.audio_token_id
    afeat = torch.randn(1, A, hidden)
    am = torch.ones(1, S, dtype=torch.long)
    cpos = torch.arange(S)
    cache0 = [torch.zeros(1, L, n_kv, hd) for _ in range(2 * n_layers)]

    with torch.no_grad():
        got = dec(ids, afeat, am, cpos, *cache0)
        # stock HF reference: same splice, standard forward
        emb = moss.model.language_model.embed_tokens(ids)
        emb = emb.masked_scatter((ids == dec.audio_token_id).unsqueeze(-1),
                                 afeat.reshape(-1, hidden))
        ref = moss.model.language_model(inputs_embeds=emb, use_cache=False,
                                        return_dict=True).last_hidden_state
        ref_logits = moss.lm_head(ref)
    e1 = float((got[0] - ref_logits).abs().max())
    print(f"parity prefill: max|Δ| = {e1:.2e}")

    # cached step: token S with past = deltas written at 0..S-1
    with torch.no_grad():
        cache1 = list(cache0)
        for i in range(2 * n_layers):
            cache1[i] = cache0[i].index_copy(1, cpos, got[1 + i])
        ids2 = torch.randint(1000, 2000, (1, 1))
        got2 = dec(ids2, torch.zeros(1, 0, hidden), torch.ones(1, 1, dtype=torch.long),
                   torch.tensor([S]), *cache1)
        emb_full = moss.model.language_model.embed_tokens(torch.cat([ids, ids2], 1))
        emb_full = emb_full.masked_scatter(
            (torch.cat([ids, ids2], 1) == dec.audio_token_id).unsqueeze(-1),
            afeat.reshape(-1, hidden))
        ref2 = moss.lm_head(moss.model.language_model(
            inputs_embeds=emb_full, use_cache=False, return_dict=True
        ).last_hidden_state[:, -1:])
    e2 = float((got2[0] - ref2).abs().max())
    print(f"parity cached step: max|Δ| = {e2:.2e}")
    ok = e1 < 1e-3 and e2 < 1e-3
    print(f"PARITY {'PASS' if ok else 'FAIL'}")
    if args.parity_only or not ok:
        return 0 if ok else 1

    # ------------------------------ export --------------------------------
    in_names = (["input_ids", "audio_features", "attention_mask", "cache_position"]
                + [f"cache_{t}_{i}" for i in range(n_layers) for t in ("key", "value")])
    out_names = (["logits"]
                 + [f"{t}_delta_{i}" for i in range(n_layers) for t in ("key", "value")])
    dyn = {"input_ids": {0: "batch", 1: "seq"},
           "audio_features": {0: "batch", 1: "n_audio_tokens"},
           "attention_mask": {0: "batch", 1: "seq"},
           "cache_position": {0: "seq"},
           "logits": {0: "batch", 1: "seq"}}
    for i in range(n_layers):
        for t in ("key", "value"):
            dyn[f"cache_{t}_{i}"] = {0: "batch"}
            dyn[f"{t}_delta_{i}"] = {0: "batch", 1: "seq"}
    print("exporting decoder.onnx (sherpa qwen3-asr interface) ...")
    torch.onnx.export(dec, (ids, afeat, am, cpos, *cache0), str(OUT / "decoder.onnx"),
                      input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    print(f"saved -> {OUT}/decoder.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
