"""Consistency checks between the YAML configs and the code that consumes them.

These guard the pipeline hand-offs: stage-2 prune targets must be
GQA-compatible with the stage-1 pruned geometry, the QAT/export RAM budget
must reflect the stage-2 geometry, and dialogue-script domains must be ones
``generate_scripts`` accepts.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from distil_vibevoice.data.dialogue_scripts import DOMAINS, generate_scripts
from distil_vibevoice.pruning.prune import _select_heads

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_data_yaml_domains_are_known() -> None:
    domains = _load("data.yaml")["dialogue_scripts"]["domains"]
    assert domains, "dialogue_scripts.domains must not be empty"
    unknown = [d for d in domains if d not in DOMAINS]
    assert not unknown, f"unknown domains in configs/data.yaml: {unknown}"
    # generate_scripts (as called by scripts/02_generate_scripts.py) accepts them.
    scripts = generate_scripts(len(domains), domains=domains, seed=0)
    assert [s.domain for s in scripts] == domains


def test_stage2_prune_targets_gqa_compatible_with_stage1() -> None:
    t1 = _load("prune_4b.yaml")["targets"]
    t2 = _load("prune_1p5b.yaml")["targets"]
    q1, kv1 = int(t1["q_heads"]), int(t1["kv_heads"])
    q2, kv2 = int(t2["q_heads"]), int(t2["kv_heads"])
    assert q1 % kv1 == 0 and q2 % kv2 == 0, "GQA needs q_heads divisible by kv_heads"
    assert q2 <= q1 and kv2 <= kv1
    group_old = q1 // kv1
    assert q2 // kv2 <= group_old, (
        f"stage-2 GQA group size {q2 // kv2} exceeds stage-1 group size "
        f"{group_old}; prune_qwen2_width would raise"
    )
    # _select_heads must accept the stage-1 -> stage-2 geometry without raising.
    q_keep, kv_keep = _select_heads(
        torch.rand(q1), torch.rand(kv1), group_old, q2, kv2
    )
    assert q_keep.numel() == q2 and kv_keep.numel() == kv2


def test_qat_export_ram_budget_matches_stage2_geometry() -> None:
    t2 = _load("prune_1p5b.yaml")["targets"]
    ram = _load("qat_export.yaml")["export"]["ram_budget"]
    assert int(ram["n_kv_heads"]) == int(t2["kv_heads"])
    assert int(ram["n_layers"]) == int(t2["layers"])
    assert int(ram["hidden"]) == int(t2["hidden"])
