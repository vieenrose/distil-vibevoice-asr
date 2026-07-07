"""Grad-enabled speech encoding for connector distillation.

VibeVoice-ASR's ``encode_speech`` wraps the acoustic/semantic tokenizer encoders
AND the connectors in a single ``torch.no_grad()`` block, so a distillation forward
pass conditions the LLM on audio features that are constants — the connectors never
receive gradients and their (pruned/reprojected) weights cannot adapt.

:func:`enable_connector_training` rebinds ``model.encode_speech`` to a variant that
keeps the encoders frozen (under ``no_grad`` — we never want to train the σ-VAE
tokenizers) but runs the connectors WITH gradients, so a normal ``model.forward``
trains the connector reprojection end-to-end. Only the non-streaming (≤ segment
duration) path is grad-enabled — training windows are short by design; longer audio
raises a clear error.
"""
from __future__ import annotations

import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["trainable_encode_speech", "enable_connector_training"]


def _resolve_dtype(model) -> "torch.dtype":
    import torch

    cfg = getattr(model, "config", None)
    td = getattr(cfg, "torch_dtype", None)
    if isinstance(td, str):
        return getattr(torch, td)
    if td is not None:
        return td
    return torch.float32


def trainable_encode_speech(
    self,
    speech_tensors: "torch.FloatTensor",
    speech_masks: "torch.BoolTensor | None" = None,
    speech_semantic_tensors: "torch.FloatTensor | None" = None,
    streaming_segment_duration: float = 60.0,
) -> "torch.Tensor":
    """Encoders frozen (no_grad); connectors run with grad. Non-streaming path only."""
    import torch

    dtype = _resolve_dtype(self)
    speech_tensors = speech_tensors.to(dtype)
    if speech_tensors.ndim == 1:
        speech_tensors = speech_tensors.unsqueeze(0)

    _batch, total_samples = speech_tensors.shape
    segment_samples = int(streaming_segment_duration * 24000)
    if total_samples > segment_samples:
        raise ValueError(
            f"trainable_encode_speech handles non-streaming audio only "
            f"(<= {streaming_segment_duration}s); got {total_samples / 24000:.1f}s. "
            f"Train on shorter windows."
        )

    # Frozen encoders: detached tokens (we never train the σ-VAE tokenizers).
    with torch.no_grad():
        ac_out = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
        audio_tokens = ac_out.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
        if speech_semantic_tensors is None:
            semantic_tokens = self.model.semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
        else:
            semantic_tokens = speech_semantic_tensors
    audio_tokens = audio_tokens.detach()
    semantic_tokens = semantic_tokens.detach()

    # Connectors WITH grad — this is what the distillation now trains.
    acoustic_features = self.model.acoustic_connector(audio_tokens)
    semantic_features = self.model.semantic_connector(semantic_tokens)

    if speech_masks is not None:
        return acoustic_features[speech_masks] + semantic_features[speech_masks]
    return acoustic_features + semantic_features


def enable_connector_training(model) -> list["torch.nn.Parameter"]:
    """Rebind encode_speech to the grad-enabled variant; freeze encoders, unfreeze connectors.

    Returns the connector parameters (to add to the optimizer). The LLM / lm_head /
    embeddings remain whatever the caller set; this only touches the audio front-end.
    """
    for name in ("acoustic_tokenizer", "semantic_tokenizer"):
        mod = getattr(model.model, name, None)
        if mod is not None:
            mod.requires_grad_(False)

    conn_params: list = []
    for name in ("acoustic_connector", "semantic_connector"):
        mod = getattr(model.model, name, None)
        if mod is not None:
            mod.requires_grad_(True)
            conn_params.extend(p for p in mod.parameters() if p.requires_grad)

    model.encode_speech = types.MethodType(trainable_encode_speech, model)
    return conn_params
