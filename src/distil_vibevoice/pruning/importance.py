"""Activation-magnitude importance scores for Minitron-style width pruning.

Collects, via forward hooks and a handful of calibration batches:

- ``hidden``   — mean ``|x|`` per residual-stream channel, aggregated over the
  inputs of every RMSNorm that reads the residual stream (each decoder layer's
  ``input_layernorm`` and ``post_attention_layernorm`` plus the final
  ``model.norm``).
- ``heads``    — per layer, per Q head: mean L2 norm of that head's attention
  output (the input of ``o_proj`` reshaped to ``[..., n_heads, head_dim]``).
- ``kv_heads`` — per layer, per KV head: sum of the grouped Q-head scores in
  that GQA group.
- ``ffn``      — per layer, per intermediate channel: mean
  ``|act_fn(gate(x)) * up(x)|`` (the input of ``down_proj``).

Works on any ``transformers`` Qwen2ForCausalLM (module layout
``model.model.layers[i].self_attn / .mlp``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass
class ImportanceScores:
    """Channel/head importance scores. All tensors are float32 on CPU.

    Attributes:
        hidden: ``[hidden_size]`` residual-stream channel scores (model-wide).
        heads: per layer, ``[num_attention_heads]`` Q-head scores.
        kv_heads: per layer, ``[num_key_value_heads]`` KV-head (GQA group) scores.
        ffn: per layer, ``[intermediate_size]`` FFN channel scores.
    """

    hidden: "torch.Tensor"
    heads: list["torch.Tensor"]
    kv_heads: list["torch.Tensor"]
    ffn: list["torch.Tensor"]


def _batch_to_inputs(batch: Any, device: str) -> dict[str, torch.Tensor]:
    """Normalize a dataloader batch into model kwargs on ``device``."""
    if isinstance(batch, torch.Tensor):
        return {"input_ids": batch.to(device)}
    if isinstance(batch, dict):
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask") and isinstance(v, torch.Tensor)
        }
        if "input_ids" not in inputs:
            raise ValueError("batch dict must contain an 'input_ids' tensor")
        return inputs
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
        return {"input_ids": batch[0].to(device)}
    raise TypeError(f"unsupported batch type for importance calibration: {type(batch)!r}")


def compute_importance(
    model,
    dataloader: Iterable[Any],
    num_batches: int = 64,
    device: str = "cuda:0",
) -> ImportanceScores:
    """Run ``num_batches`` calibration batches and collect importance scores.

    Args:
        model: a ``transformers`` Qwen2ForCausalLM (or same module layout).
        dataloader: yields tensors of input ids, or dicts with ``input_ids``
            (and optionally ``attention_mask``).
        num_batches: maximum number of batches to consume.
        device: device to run calibration on; the model is moved there.

    Returns:
        :class:`ImportanceScores` with float32 CPU tensors.
    """
    model = model.to(device)
    was_training = model.training
    model.eval()

    cfg = model.config
    n_layers: int = cfg.num_hidden_layers
    hidden_size: int = cfg.hidden_size
    n_q: int = cfg.num_attention_heads
    n_kv: int = cfg.num_key_value_heads
    intermediate: int = cfg.intermediate_size
    head_dim: int = getattr(cfg, "head_dim", None) or hidden_size // n_q
    group_size = n_q // n_kv

    hidden_acc = torch.zeros(hidden_size, dtype=torch.float32)
    hidden_n = 0
    head_acc = [torch.zeros(n_q, dtype=torch.float32) for _ in range(n_layers)]
    head_n = [0] * n_layers
    ffn_acc = [torch.zeros(intermediate, dtype=torch.float32) for _ in range(n_layers)]
    ffn_n = [0] * n_layers

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hidden_hook(module: torch.nn.Module, args: tuple) -> None:
        nonlocal hidden_n
        x = args[0]
        hidden_acc.add_(x.detach().abs().float().mean(dim=tuple(range(x.dim() - 1))).cpu())
        hidden_n += 1

    def _make_head_hook(layer_idx: int):
        def hook(module: torch.nn.Module, args: tuple) -> None:
            x = args[0].detach()  # [..., n_q * head_dim], input of o_proj
            per_head = x.reshape(-1, n_q, head_dim).float().norm(dim=-1).mean(dim=0)
            head_acc[layer_idx].add_(per_head.cpu())
            head_n[layer_idx] += 1

        return hook

    def _make_ffn_hook(layer_idx: int):
        def hook(module: torch.nn.Module, args: tuple) -> None:
            x = args[0].detach()  # [..., intermediate], input of down_proj
            ffn_acc[layer_idx].add_(x.reshape(-1, intermediate).abs().float().mean(dim=0).cpu())
            ffn_n[layer_idx] += 1

        return hook

    decoder = model.model  # Qwen2Model
    for i, layer in enumerate(decoder.layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(_hidden_hook))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(_hidden_hook))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(_make_head_hook(i)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(_make_ffn_hook(i)))
    handles.append(decoder.norm.register_forward_pre_hook(_hidden_hook))

    seen = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                if seen >= num_batches:
                    break
                inputs = _batch_to_inputs(batch, device)
                model(**inputs, use_cache=False)
                seen += 1
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    if seen == 0:
        raise ValueError("dataloader yielded no batches; cannot compute importance")

    hidden = hidden_acc / max(hidden_n, 1)
    heads = [acc / max(n, 1) for acc, n in zip(head_acc, head_n)]
    ffn = [acc / max(n, 1) for acc, n in zip(ffn_acc, ffn_n)]
    # KV-head score = aggregate of its GQA group's Q-head scores.
    kv_heads = [h.reshape(n_kv, group_size).sum(dim=-1) for h in heads]

    return ImportanceScores(hidden=hidden, heads=heads, kv_heads=kv_heads, ffn=ffn)
