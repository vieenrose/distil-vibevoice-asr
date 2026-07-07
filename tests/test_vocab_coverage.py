"""CPU-only tests for vocab-coverage analysis and trimmed-vocab distillation.

No network, no pretrained tokenizer: a tiny dict-based ``FakeTokenizer`` stub
stands in for a real one.
"""
from __future__ import annotations

import math
import string
import tempfile
from pathlib import Path

import torch

from distil_vibevoice.data.manifest import MeetingRecord, Segment, write_manifest
from distil_vibevoice.data.vocab_coverage import (
    analyze_manifest_vocab,
    build_kept_vocab,
)
from distil_vibevoice.distill.losses import distill_loss

torch.manual_seed(0)


# ----------------------------------------------------------------------
# Fake tokenizer: char -> id over printable ASCII, plus fake byte + special
# tokens that are never emitted by ``encode`` (they only exist in the vocab).
# ----------------------------------------------------------------------
class FakeTokenizer:
    def __init__(self) -> None:
        chars = string.ascii_letters + string.digits + string.punctuation + " "
        self.vocab: dict[str, int] = {c: i for i, c in enumerate(chars)}
        base = len(self.vocab)
        self.byte_ids: list[int] = []
        for b in range(8):  # a handful of fake <0xNN> byte-fallback tokens
            self.vocab[f"<0x{b:02X}>"] = base + b
            self.byte_ids.append(base + b)
        base = len(self.vocab)
        self.special = {"<pad>": base, "<eos>": base + 1}
        self.vocab.update(self.special)
        self.all_special_ids = list(self.special.values())
        self._size = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def vocab_size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size


def _write_manifest(tmp: Path, texts: list[str]) -> str:
    records = [
        MeetingRecord(
            audio_path=f"a{i}.wav",
            duration_s=1.0,
            sample_rate=16000,
            language="en",
            source="test",
            split="train",
            segments=[Segment(start=0.0, end=1.0, speaker="0", text=t)],
        )
        for i, t in enumerate(texts)
    ]
    path = tmp / "manifest.jsonl"
    write_manifest(records, path)
    return str(path)


# ----------------------------------------------------------------------
# build_kept_vocab
# ----------------------------------------------------------------------
def test_coverage_is_one_at_min_count_one() -> None:
    tok = FakeTokenizer()
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), ["hello world", "another meeting line"])
        out = build_kept_vocab([mpath], tok, min_count=1)
    assert out["coverage"] == 1.0
    assert out["dropped_high_freq"] == []
    assert out["n_kept"] <= out["n_total"]


def test_byte_and_special_ids_always_kept_even_if_unused() -> None:
    tok = FakeTokenizer()
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), ["hi"])
        out = build_kept_vocab([mpath], tok, min_count=1)
    kept = set(out["kept_ids"])
    # None of these are producible by encode(), yet all must be retained.
    for bid in tok.byte_ids:
        assert bid in kept
    for sid in tok.all_special_ids:
        assert sid in kept


def test_include_bytes_false_drops_byte_tokens() -> None:
    tok = FakeTokenizer()
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), ["hi"])
        out = build_kept_vocab([mpath], tok, min_count=1, include_bytes=False)
    kept = set(out["kept_ids"])
    assert not any(bid in kept for bid in tok.byte_ids)
    # Specials are still force-kept regardless of include_bytes.
    for sid in tok.all_special_ids:
        assert sid in kept


def test_raising_min_count_drops_rare_ids_and_lowers_coverage() -> None:
    tok = FakeTokenizer()
    # 'z' appears once total; common chars appear many times.
    texts = ["aaaa aaaa", "aaaa aaaa", "z"]
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), texts)
        base = build_kept_vocab([mpath], tok, min_count=1)
        strict = build_kept_vocab([mpath], tok, min_count=3)
    assert base["coverage"] == 1.0
    assert strict["coverage"] < 1.0
    assert strict["coverage"] > 0.0
    assert strict["n_kept"] < base["n_kept"]
    # 'z' (count 1 < 3) is dropped and surfaced with its count.
    z_id = tok.vocab["z"]
    dropped_ids = {tid for tid, _ in strict["dropped_high_freq"]}
    assert z_id in dropped_ids
    assert dict(strict["dropped_high_freq"])[z_id] == 1


def test_extra_ids_honored() -> None:
    tok = FakeTokenizer()
    unused = tok.vocab["~"]  # a char id not present in the target text
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), ["abc"])
        without = build_kept_vocab([mpath], tok, min_count=1)
        withx = build_kept_vocab([mpath], tok, min_count=1, extra_ids={unused})
    assert unused not in set(without["kept_ids"])
    assert unused in set(withx["kept_ids"])


def test_analyze_reports_savings() -> None:
    tok = FakeTokenizer()
    with tempfile.TemporaryDirectory() as d:
        mpath = _write_manifest(Path(d), ["short line"])
        out = analyze_manifest_vocab([mpath], tok, min_count=1, hidden_size=1536)
    sav = out["projected_savings"]
    rows_dropped = out["n_total"] - out["n_kept"]
    assert sav["rows_dropped"] == rows_dropped
    assert sav["params_saved"] == rows_dropped * 1536 * 1  # tied by default
    assert sav["ram_saved_bytes"] == sav["params_saved"] * 2


# ----------------------------------------------------------------------
# distill_loss with kept_vocab_ids
# ----------------------------------------------------------------------
def test_trimmed_vocab_loss_is_finite() -> None:
    V, m = 512, 64
    kept = torch.arange(0, 2 * m, 2)  # 64 even ids in [0, 128)
    student = torch.randn(2, 6, m)
    teacher = torch.randn(2, 6, V)
    labels = torch.randint(0, V, (2, 6))
    out = distill_loss(
        student, teacher, labels, kept_vocab_ids=kept, T=2.0,
        w_kl=1.0, w_ce=1.0, w_hidden=0.0,
    )
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["kl"])
    assert float(out["kl"]) >= 0.0
    assert torch.isfinite(out["ce"])


def test_labels_outside_kept_set_are_ignored() -> None:
    # Kept set excludes id 0; a label of 0 must be ignored (no CE contribution).
    V, m = 16, 4
    kept = torch.tensor([1, 3, 5, 7])
    student = torch.zeros(1, 2, m)
    teacher = torch.zeros(1, 2, V)
    labels = torch.tensor([[0, 3]])  # first not in kept, second maps to col 1
    out = distill_loss(
        student, teacher, labels, kept_vocab_ids=kept,
        w_kl=0.0, w_ce=1.0, w_hidden=0.0, T=1.0,
    )
    # Only the second position is valid; uniform logits -> CE = log(m).
    assert math.isclose(float(out["ce"]), math.log(m), rel_tol=1e-5)


def test_identity_when_kept_is_all_ids() -> None:
    V = 32
    student = torch.randn(2, 5, V)
    teacher = torch.randn(2, 5, V)
    labels = torch.randint(0, V, (2, 5))
    dense = distill_loss(
        student, teacher, labels, T=1.5, w_kl=1.0, w_ce=1.0, w_hidden=0.0
    )
    trim = distill_loss(
        student, teacher, labels, kept_vocab_ids=torch.arange(V),
        T=1.5, w_kl=1.0, w_ce=1.0, w_hidden=0.0,
    )
    assert math.isclose(float(dense["loss"]), float(trim["loss"]), rel_tol=1e-6)
    assert math.isclose(float(dense["kl"]), float(trim["kl"]), rel_tol=1e-6)
    assert math.isclose(float(dense["ce"]), float(trim["ce"]), rel_tol=1e-6)
