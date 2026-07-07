"""Peak-RAM estimation for the on-device (6 GB phone) deployment.

Component model (decimal GB = 1e9 bytes):

- ``weights``:      LLM body, ``params_b * 1e9 * bytes_per_param(quant)``.
  int4 uses 0.5625 B/param (q4_k average incl. scales/mins), int8 uses
  1.0625 B/param (per-group scales), fp16/bf16 use 2.
- ``kv_cache``:     ``n_layers * n_kv_heads * head_dim * 2 (K,V) *
  bytes(kv_dtype) * context_tokens``.
- ``embeddings``:   int8 embedding table, ``vocab * hidden * 1`` byte.
  When ``tied=True`` this term is zeroed: a tied model shares one
  ``embed_tokens``/``lm_head`` matrix that is already counted once inside
  ``params_b``, so adding a separate table would double-count it.
- ``activations``:  transient working set, ``6 * hidden * context_tokens``
  fp16 elements (residual + attention + MLP buffers, small term).
- ``encoder``:      frozen acoustic+semantic encoders,
  ``encoder_params_b * 1e9 * bytes_per_param(encoder_quant)``.
- ``overhead``:     runtime/allocator/OS slack, flat ``overhead_gb``.

CLI: ``python -m distil_vibevoice.runtime.ram_budget --params-b 1.5 --context 8192``
"""
from __future__ import annotations

import argparse

__all__ = ["estimate_ram", "pretty_print", "main"]

_GB = 1e9

#: Average bytes per parameter for weight quantization schemes.
_WEIGHT_BYTES: dict[str, float] = {
    "int4": 0.5625,  # q4_k average incl. group scales/mins
    "int8": 1.0625,  # int8 + per-group scales
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
}

#: Bytes per element for KV-cache dtypes.
_KV_BYTES: dict[str, float] = {
    "int8": 1.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
}

#: Bytes per fp16 activation element (working-set term).
_ACT_BYTES = 2.0
#: Activation working set: ~6 live [hidden] vectors per context token.
_ACT_VECTORS_PER_TOKEN = 6


def _bytes_per_param(quant: str, table: dict[str, float], what: str) -> float:
    try:
        return table[quant.lower()]
    except KeyError:
        raise ValueError(f"unknown {what} {quant!r}; choose one of {sorted(table)}") from None


def estimate_ram(
    params_b: float,
    quant: str = "int4",
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    context_tokens: int = 8192,
    kv_dtype: str = "fp16",
    encoder_params_b: float = 0.7,
    encoder_quant: str = "int4",
    vocab: int = 152064,
    hidden: int = 1536,
    overhead_gb: float = 0.4,
    tied: bool = False,
) -> dict:
    """Estimate peak inference RAM in GB per component.

    Returns a dict with keys ``weights_gb``, ``kv_cache_gb``,
    ``embeddings_gb``, ``activations_gb``, ``encoder_gb``, ``overhead_gb``
    and their sum ``total_gb``.
    """
    weights = params_b * 1e9 * _bytes_per_param(quant, _WEIGHT_BYTES, "quant")
    kv_cache = (
        n_layers
        * n_kv_heads
        * head_dim
        * 2  # K and V
        * _bytes_per_param(kv_dtype, _KV_BYTES, "kv_dtype")
        * context_tokens
    )
    # int8 embedding table. When tied, embed_tokens == lm_head is a single
    # shared matrix already counted once inside params_b, so counting a
    # separate table here would double-count it.
    embeddings = 0.0 if tied else float(vocab) * hidden * 1.0
    activations = _ACT_VECTORS_PER_TOKEN * hidden * context_tokens * _ACT_BYTES
    encoder = encoder_params_b * 1e9 * _bytes_per_param(
        encoder_quant, _WEIGHT_BYTES, "encoder_quant"
    )

    report = {
        "weights_gb": weights / _GB,
        "kv_cache_gb": kv_cache / _GB,
        "embeddings_gb": embeddings / _GB,
        "activations_gb": activations / _GB,
        "encoder_gb": encoder / _GB,
        "overhead_gb": float(overhead_gb),
    }
    report["total_gb"] = sum(report.values())
    return report


def pretty_print(report: dict) -> str:
    """Render an estimate_ram() report as an aligned text table."""
    labels = {
        "weights_gb": "LLM weights",
        "kv_cache_gb": "KV cache",
        "embeddings_gb": "Embeddings (int8)",
        "activations_gb": "Activations",
        "encoder_gb": "Audio encoders",
        "overhead_gb": "Runtime overhead",
        "total_gb": "TOTAL",
    }
    width = max(len(v) for v in labels.values())
    lines = ["Peak RAM estimate", "-" * (width + 12)]
    for key, label in labels.items():
        if key not in report:
            continue
        if key == "total_gb":
            lines.append("-" * (width + 12))
        lines.append(f"{label:<{width}}  {report[key]:7.3f} GB")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point; prints the table and returns the report dict."""
    parser = argparse.ArgumentParser(
        prog="python -m distil_vibevoice.runtime.ram_budget",
        description="Estimate on-device peak RAM for the distilled student.",
    )
    parser.add_argument("--params-b", type=float, default=1.5, help="LLM params in billions")
    parser.add_argument("--quant", default="int4", help="LLM weight quant (int4/int8/fp16/fp32)")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", dest="context_tokens", type=int, default=8192)
    parser.add_argument("--kv-dtype", default="fp16", help="KV cache dtype (int8/fp16/fp32)")
    parser.add_argument("--encoder-params-b", type=float, default=0.7)
    parser.add_argument("--encoder-quant", default="int4")
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--overhead-gb", type=float, default=0.4)
    parser.add_argument(
        "--tied",
        action="store_true",
        help="tied embed_tokens/lm_head: zero the separate embedding term (already in params_b)",
    )
    args = parser.parse_args(argv)

    report = estimate_ram(
        params_b=args.params_b,
        quant=args.quant,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        context_tokens=args.context_tokens,
        kv_dtype=args.kv_dtype,
        encoder_params_b=args.encoder_params_b,
        encoder_quant=args.encoder_quant,
        vocab=args.vocab,
        hidden=args.hidden,
        overhead_gb=args.overhead_gb,
        tied=args.tied,
    )
    print(pretty_print(report))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
