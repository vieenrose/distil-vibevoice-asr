"""Straight-through int4 fake-quantization matching onnxruntime MatMulNBits.

Deployment quantizes decoder Linear weights to 4-bit, block_size=32, symmetric
(the MatMul4BitsQuantizer format). To make q4 near-lossless we QAT with the
SAME granularity so the trained weights are robust to that exact rounding.

quantize_weight(W): per-(output row, input block of 32) symmetric int4,
                    dequantized; gradient passes straight through (STE).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def fake_quant_int4(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    """STE symmetric int4 (levels -7..7) per row per input-block of `block`."""
    out_f, in_f = w.shape
    pad = (block - in_f % block) % block
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    wv = w.view(out_f, -1, block)                      # [out, nblk, block]
    absmax = wv.abs().amax(dim=-1, keepdim=True)       # [out, nblk, 1]
    scale = (absmax / 7.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(wv / scale), -7, 7)
    dq = (q * scale).view(out_f, -1)[:, :in_f]
    # straight-through: forward dq, backward gradient of identity
    return w[:, :in_f] + (dq - w[:, :in_f]).detach()


class QATLinear(nn.Module):
    """Wraps an nn.Linear; fake-quantizes the weight on every forward."""

    def __init__(self, lin: nn.Linear, block: int = 32):
        super().__init__()
        self.lin = lin
        self.block = block
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lin.weight
        if self.enabled:
            w = fake_quant_int4(w, self.block)
        return torch.nn.functional.linear(x, w, self.lin.bias)


# Linear submodule name fragments that become MatMulNBits in the q4 export.
_Q4_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj", "lm_head")


def wrap_decoder_linears(model, targets=_Q4_TARGETS, exclude=()):
    """Replace target Linear modules with QATLinear in place. Returns the list
    of wrapped qualified names (order = model.named_modules order)."""
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear)
                    and any(t in child_name for t in targets)
                    and full not in exclude):
                setattr(module, child_name, QATLinear(child))
                wrapped.append(full)
    return wrapped


def set_fakequant(model, enabled: bool):
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.enabled = enabled
