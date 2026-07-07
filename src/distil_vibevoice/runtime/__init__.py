"""On-device runtime: chunked inference, speaker stitching, RAM budget, mobile export.

Submodules are imported lazily so that light-weight tools (e.g.
``python -m distil_vibevoice.runtime.ram_budget``) do not pull in audio or
model dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ChunkedTranscriber": ("distil_vibevoice.runtime.chunked_inference", "ChunkedTranscriber"),
    "match_speakers": ("distil_vibevoice.runtime.speaker_stitch", "match_speakers"),
    "stitch": ("distil_vibevoice.runtime.speaker_stitch", "stitch"),
    "BaseEmbedder": ("distil_vibevoice.runtime.embeddings", "BaseEmbedder"),
    "MfccStatsEmbedder": ("distil_vibevoice.runtime.embeddings", "MfccStatsEmbedder"),
    "OnnxSpeakerEmbedder": ("distil_vibevoice.runtime.embeddings", "OnnxSpeakerEmbedder"),
    "load_embedder": ("distil_vibevoice.runtime.embeddings", "load_embedder"),
    "SpeakerProfile": ("distil_vibevoice.runtime.speaker_registry", "SpeakerProfile"),
    "SpeakerRegistry": ("distil_vibevoice.runtime.speaker_registry", "SpeakerRegistry"),
    "consolidate": ("distil_vibevoice.runtime.consolidate", "consolidate"),
    "estimate_ram": ("distil_vibevoice.runtime.ram_budget", "estimate_ram"),
    "pretty_print": ("distil_vibevoice.runtime.ram_budget", "pretty_print"),
    "gguf_conversion_plan": ("distil_vibevoice.runtime.export_mobile", "gguf_conversion_plan"),
    "write_gguf_conversion_script": ("distil_vibevoice.runtime.export_mobile", "write_gguf_conversion_script"),
    "export_encoder_onnx": ("distil_vibevoice.runtime.export_mobile", "export_encoder_onnx"),
    "quantize_int4": ("distil_vibevoice.runtime.export_mobile", "quantize_int4"),
    "export_executorch": ("distil_vibevoice.runtime.export_mobile", "export_executorch"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
