"""Mobile export paths for the distilled student.

Two documented routes:

(a) **GGUF / llama.cpp** — the pruned student LLM is a *standard*
    ``Qwen2ForCausalLM`` (width pruning keeps the architecture), so the stock
    ``convert_hf_to_gguf.py`` converter works unmodified.
    :func:`gguf_conversion_plan` emits the exact command sequence and
    :func:`write_gguf_conversion_script` materializes it as a bash script.
    The frozen acoustic/semantic encoders are NOT part of the GGUF graph;
    export them separately with :func:`export_encoder_onnx`.

(b) **ExecuTorch / torchao int4** — :func:`quantize_int4` applies torchao
    int4 weight-only quantization in-place; :func:`export_executorch` lowers
    a module through ``torch.export`` -> ExecuTorch edge dialect to a
    ``.pte`` file.

Fusing the encoders + connector + decoder into ONE mobile graph requires the
real trained student checkpoint and is deliberately left as a documented
:class:`NotImplementedError` in :func:`export_full_asr_pipeline`.
"""
from __future__ import annotations

import shlex
import stat
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "gguf_conversion_plan",
    "write_gguf_conversion_script",
    "export_encoder_onnx",
    "quantize_int4",
    "export_executorch",
    "export_full_asr_pipeline",
]

#: Total waveform downsampling of the VibeVoice acoustic/semantic encoders
#: (ratios [8, 5, 5, 4, 2, 2] -> 3200x @ 24 kHz = 7.5 latents/sec).
ENCODER_HOP = 3200


# --------------------------------------------------------------------------- #
# (a) GGUF via llama.cpp
# --------------------------------------------------------------------------- #

def gguf_conversion_plan(
    model_dir: str,
    out_dir: str,
    quant: str = "Q4_K_M",
    llama_cpp_dir: str = "third_party/llama.cpp",
) -> list[str]:
    """Command plan (shell lines, comments included) to convert the student to GGUF.

    ``model_dir`` must hold a saved HF checkpoint of the pruned student LLM
    (``student.save_pretrained(model_dir)`` + tokenizer). Works because the
    width-pruned student remains an architecture-standard Qwen2.
    """
    model_q = shlex.quote(str(model_dir))
    out = str(out_dir).rstrip("/")
    llama = str(llama_cpp_dir).rstrip("/")
    f16 = shlex.quote(f"{out}/student-f16.gguf")
    q_file = shlex.quote(f"{out}/student-{quant.lower()}.gguf")
    return [
        f"mkdir -p {shlex.quote(out)}",
        "# 1) HF checkpoint -> GGUF fp16. The pruned student is a standard",
        "#    Qwen2ForCausalLM, so the stock converter needs no patches.",
        f"python {llama}/convert_hf_to_gguf.py {model_q} --outfile {f16} --outtype f16",
        f"# 2) int4 K-quant ({quant}, ~0.5625 B/param incl. scales).",
        f"{llama}/build/bin/llama-quantize {f16} {q_file} {quant}",
        "# 3) Smoke test (text-only; audio latents enter via the ONNX encoders",
        "#    + connector at the app layer, injected as embeddings).",
        f"{llama}/build/bin/llama-cli -m {q_file} -p 'ping' -n 16",
        "# NOTE: export the frozen acoustic/semantic encoders separately with",
        "#       distil_vibevoice.runtime.export_mobile.export_encoder_onnx().",
    ]


def write_gguf_conversion_script(
    model_dir: str,
    out_dir: str,
    script_path: str,
    quant: str = "Q4_K_M",
    llama_cpp_dir: str = "third_party/llama.cpp",
) -> Path:
    """Write the GGUF conversion plan as an executable bash script; return its path."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines += gguf_conversion_plan(model_dir, out_dir, quant=quant, llama_cpp_dir=llama_cpp_dir)
    path = Path(script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --------------------------------------------------------------------------- #
# Encoder -> ONNX
# --------------------------------------------------------------------------- #

def export_encoder_onnx(
    encoder: "Any",
    out_path: str,
    sample_rate: int = 24000,
    example_seconds: float = 2.0,
    opset: int = 17,
) -> str:
    """Export a waveform encoder (``(B, 1, T) -> latents``) to ONNX.

    Works for the frozen VibeVoice acoustic/semantic conv encoders (or any
    ``torch.nn.Module`` taking a mono waveform tensor). The dummy input length
    is rounded to a multiple of the 3200x hop so every conv stage divides
    evenly; batch and time axes are exported as dynamic.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("torch is required for ONNX export") from e

    n_samples = max(ENCODER_HOP, int(sample_rate * example_seconds) // ENCODER_HOP * ENCODER_HOP)
    dummy = torch.randn(1, 1, n_samples)
    encoder = encoder.eval()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            # dynamo=False: the legacy TorchScript exporter matches the
            # dynamic_axes API and needs no onnxscript install.
            torch.onnx.export(
                encoder,
                (dummy,),
                str(out),
                input_names=["waveform"],
                output_names=["latents"],
                dynamic_axes={"waveform": {0: "batch", 2: "time"}, "latents": {0: "batch"}},
                opset_version=opset,
                dynamo=False,
            )
    except (ModuleNotFoundError, torch.onnx.OnnxExporterError) as e:
        # pragma: no cover - environment dependent
        if isinstance(e, torch.onnx.OnnxExporterError) and "not installed" not in str(e):
            raise
        raise ImportError(
            f"ONNX export needs an optional dependency ({e}): "
            "pip install onnx (and onnxscript for the dynamo exporter)"
        ) from e
    return str(out)


# --------------------------------------------------------------------------- #
# (b) torchao int4 + ExecuTorch
# --------------------------------------------------------------------------- #

def quantize_int4(model: "Any", group_size: int = 128) -> "Any":
    """Apply torchao int4 weight-only quantization in-place; returns the model.

    Uses ``Int4WeightOnlyConfig`` (grouped int4, tinygemm-style packing).
    TODO-verify on the real student: torchao's default int4 kernels expect
    bf16 weights and a CUDA device for the packing step; for a pure-CPU export
    pass a CPU-layout config per current torchao docs.
    """
    try:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
    except ImportError as e:
        raise ImportError(
            "torchao is required for int4 quantization: pip install torchao"
        ) from e
    quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
    return model


def export_executorch(
    model: "Any",
    out_path: str,
    example_inputs: Sequence["Any"] | None = None,
) -> str:
    """Lower ``model`` through torch.export -> ExecuTorch and write a ``.pte``.

    ``example_inputs`` must be a tuple of example tensors matching the model's
    forward signature; defaults to a ``(1, 8)`` int64 token-id tensor, which
    suits a bare ``Qwen2ForCausalLM``-style ``forward(input_ids)``. For a
    production decoder you additionally want static KV-cache rewriting
    (``transformers`` ``StaticCache`` or executorch's llama export recipes)
    before lowering — see :func:`export_full_asr_pipeline`.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("torch is required for ExecuTorch export") from e
    try:
        from executorch.exir import to_edge
    except ImportError as e:
        raise ImportError(
            "executorch is required: pip install executorch "
            "(see https://pytorch.org/executorch for platform wheels)"
        ) from e

    if example_inputs is None:
        example_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long),)
    exported = torch.export.export(model.eval(), tuple(example_inputs))
    program = to_edge(exported).to_executorch()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    return str(out)


def export_full_asr_pipeline(student_dir: str, out_dir: str, quant: str = "int4") -> None:
    """Fuse encoders + connector + decoder into one mobile bundle. NOT IMPLEMENTED.

    This genuinely needs the real trained student checkpoint (weights are
    still training / downloading), because the export must:

    1. load the VibeVoice student with its acoustic+semantic encoders and the
       width-pruned connectors (channel index sets from pruning.prune);
    2. trace the speech path (waveform -> 7.5 Hz latents -> connector ->
       ``inputs_embeds`` splice at the ``<speech_start>``/pad positions),
       which depends on the concrete processor tensor layout;
    3. rewrite the decoder with a static KV cache and per-chunk sequence
       lengths before ``torch.export`` will accept it.

    Until then, use the two verified piecewise paths:
    ``export_encoder_onnx()`` for the frozen encoders +
    ``gguf_conversion_plan()`` (llama.cpp) or ``quantize_int4()`` +
    ``export_executorch()`` for the LLM body, and splice latents at runtime.
    """
    raise NotImplementedError(
        "export_full_asr_pipeline requires the real trained student checkpoint "
        f"(expected under {student_dir!r}). Export piecewise for now: "
        "export_encoder_onnx() for the encoders and gguf_conversion_plan() / "
        "quantize_int4()+export_executorch() for the LLM body; see docstring."
    )
