"""Tests for distil_vibevoice.runtime.ram_budget (pure python, no deps)."""
from __future__ import annotations

import pytest

from distil_vibevoice.runtime.ram_budget import estimate_ram, main, pretty_print


def test_1p5b_int4_8k_total_within_mobile_budget():
    report = estimate_ram(params_b=1.5, quant="int4", context_tokens=8192)
    assert 1.8 <= report["total_gb"] <= 2.8


def test_total_is_sum_of_components():
    report = estimate_ram(params_b=1.5)
    parts = sum(v for k, v in report.items() if k != "total_gb")
    assert report["total_gb"] == pytest.approx(parts)
    for value in report.values():
        assert value >= 0.0


def test_kv_cache_scales_linearly_with_context():
    base = estimate_ram(params_b=1.5, context_tokens=4096)["kv_cache_gb"]
    double = estimate_ram(params_b=1.5, context_tokens=8192)["kv_cache_gb"]
    quad = estimate_ram(params_b=1.5, context_tokens=16384)["kv_cache_gb"]
    assert double == pytest.approx(2.0 * base)
    assert quad == pytest.approx(4.0 * base)


def test_fp16_weights_about_twice_int8():
    fp16 = estimate_ram(params_b=1.5, quant="fp16")["weights_gb"]
    int8 = estimate_ram(params_b=1.5, quant="int8")["weights_gb"]
    ratio = fp16 / int8
    assert 1.7 < ratio < 2.0
    assert ratio == pytest.approx(2.0 / 1.0625)


def test_int8_kv_halves_kv_cache_vs_fp16():
    fp16 = estimate_ram(params_b=1.5, kv_dtype="fp16")["kv_cache_gb"]
    int8 = estimate_ram(params_b=1.5, kv_dtype="int8")["kv_cache_gb"]
    assert fp16 == pytest.approx(2.0 * int8)


def test_tied_zeroes_embedding_term_and_lowers_total():
    untied = estimate_ram(params_b=1.54, quant="int4", context_tokens=8192)
    tied = estimate_ram(params_b=1.54, quant="int4", context_tokens=8192, tied=True)
    # tied model shares embed/lm_head, already inside params_b -> no separate table
    assert tied["embeddings_gb"] == 0.0
    assert untied["embeddings_gb"] > 0.0
    # every other component is unchanged
    for key in ("weights_gb", "kv_cache_gb", "activations_gb", "encoder_gb", "overhead_gb"):
        assert tied[key] == pytest.approx(untied[key])
    # total drops by exactly the removed embedding table
    assert tied["total_gb"] == pytest.approx(untied["total_gb"] - untied["embeddings_gb"])
    assert tied["total_gb"] == pytest.approx(sum(v for k, v in tied.items() if k != "total_gb"))


def test_tied_flag_default_false_is_backward_compatible():
    assert estimate_ram(params_b=1.5)["embeddings_gb"] == estimate_ram(params_b=1.5, tied=False)["embeddings_gb"]


def test_cli_tied_flag_zeroes_embeddings():
    assert main(["--params-b", "1.54", "--tied"])["embeddings_gb"] == 0.0
    assert main(["--params-b", "1.54"])["embeddings_gb"] > 0.0


def test_unknown_quant_raises():
    with pytest.raises(ValueError):
        estimate_ram(params_b=1.5, quant="int3")
    with pytest.raises(ValueError):
        estimate_ram(params_b=1.5, kv_dtype="fp8")


def test_pretty_print_lists_components_and_total():
    text = pretty_print(estimate_ram(params_b=1.5))
    assert "TOTAL" in text
    assert "KV cache" in text
    assert "GB" in text


def test_cli_main_parses_args_and_returns_report():
    report = main(["--params-b", "1.5", "--context", "8192"])
    assert 1.8 <= report["total_gb"] <= 2.8
    bigger = main(["--params-b", "1.5", "--context", "16384"])
    assert bigger["kv_cache_gb"] == pytest.approx(2.0 * report["kv_cache_gb"])
