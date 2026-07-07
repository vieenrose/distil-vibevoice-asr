"""Shared pytest fixtures: tiny CPU-only models and sample manifests.

No GPU, no network, no downloads. Model fixtures build randomly initialized
Qwen2 models from a config so pruning/distillation tests run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tiny_qwen2():
    """Randomly initialized tiny Qwen2ForCausalLM on CPU (no download)."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    config = transformers.Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=512,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture()
def tiny_qwen2_config():
    """The Qwen2Config matching ``tiny_qwen2``, for tests that only need shapes."""
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=512,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )


@pytest.fixture()
def sample_segments():
    """A small two-speaker zh-TW/en code-switched segment list."""
    from distil_vibevoice.data.manifest import Segment

    return [
        Segment(start=0.0, end=3.2, speaker="0", text="大家好，我們開始今天的會議。"),
        Segment(start=3.2, end=7.8, speaker="1", text="好的，先看一下 roadmap 的 milestone。"),
        Segment(start=7.8, end=12.5, speaker="0", text="這個 feature 下週要 release，測試進度如何？"),
        Segment(start=12.5, end=15.0, speaker="1", text="Integration tests are done, 剩下 UI 的部分。"),
    ]


@pytest.fixture()
def sample_record(sample_segments, tmp_path: Path):
    """A MeetingRecord pointing at a (nonexistent) wav under tmp_path."""
    from distil_vibevoice.data.manifest import MeetingRecord

    return MeetingRecord(
        audio_path=str(tmp_path / "meeting_000.wav"),
        duration_s=15.0,
        sample_rate=24000,
        language="zh-TW-en",
        source="unit-test",
        split="train",
        segments=sample_segments,
        meta={"mean_logprob": -0.21},
    )


@pytest.fixture()
def tmp_manifest(sample_record, tmp_path: Path) -> Path:
    """A JSONL manifest file with one MeetingRecord, written to tmp_path."""
    from distil_vibevoice.data.manifest import write_manifest

    path = tmp_path / "manifest.jsonl"
    write_manifest([sample_record], path)
    return path
