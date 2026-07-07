"""CPU-only tests for distillation losses and the trainer loop.

No network, no GPU, no pretrained weights: models are tiny randomly
initialized torch modules exposing the HF causal-LM output interface
(``.logits`` / ``.hidden_states`` / ``.config``).
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from distil_vibevoice.distill.losses import (
    build_token_weights,
    default_layer_map,
    distill_loss,
)
from distil_vibevoice.distill.trainer import DistillTrainer

torch.manual_seed(0)


# ----------------------------------------------------------------------
# distill_loss
# ----------------------------------------------------------------------
def test_kl_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 7, 33)
    labels = torch.randint(0, 33, (2, 7))
    out = distill_loss(logits, logits.clone(), labels)
    assert float(out["kl"]) == abs(float(out["kl"]))  # non-negative
    assert float(out["kl"]) < 1e-6
    assert torch.isfinite(out["loss"])


def test_kl_nonnegative_and_smaller_when_matching() -> None:
    teacher = torch.randn(2, 5, 17)
    student_far = torch.randn(2, 5, 17)
    labels = torch.randint(0, 17, (2, 5))
    far = distill_loss(student_far, teacher, labels, T=1.0, w_kl=1.0, w_ce=0.0, w_hidden=0.0)
    near = distill_loss(teacher.clone(), teacher, labels, T=1.0, w_kl=1.0, w_ce=0.0, w_hidden=0.0)
    assert float(far["kl"]) >= 0.0
    assert float(near["kl"]) >= 0.0
    assert float(near["kl"]) < float(far["kl"])
    # With w_kl=1, w_ce=0, w_hidden=0 the total loss is exactly the KL term.
    assert math.isclose(float(far["loss"]), float(far["kl"]), rel_tol=1e-6)


def test_ce_matches_manual_cross_entropy() -> None:
    logits = torch.randn(1, 4, 11)
    labels = torch.randint(0, 11, (1, 4))
    out = distill_loss(logits, logits, labels, w_kl=0.0, w_ce=1.0, w_hidden=0.0, T=1.0)
    manual = nn.functional.cross_entropy(logits.view(-1, 11), labels.view(-1))
    assert math.isclose(float(out["ce"]), float(manual), rel_tol=1e-5)
    assert math.isclose(float(out["loss"]), float(manual), rel_tol=1e-5)


def test_masked_positions_contribute_zero() -> None:
    torch.manual_seed(1)
    logits_s = torch.randn(1, 6, 13)
    logits_t = torch.randn(1, 6, 13)
    labels = torch.randint(0, 13, (1, 6))
    labels[0, 2] = -100
    labels[0, 5] = -100

    base = distill_loss(logits_s, logits_t, labels)

    # Garbage at masked positions must not change any loss component.
    s2, t2 = logits_s.clone(), logits_t.clone()
    s2[0, 2] = 100.0
    t2[0, 5] = -50.0
    perturbed = distill_loss(s2, t2, labels)
    for key in ("loss", "kl", "ce"):
        assert math.isclose(float(base[key]), float(perturbed[key]), rel_tol=1e-5, abs_tol=1e-7)


def test_token_weights_upweight_tagged_positions_only() -> None:
    labels = torch.tensor([[5, 9, 3, -100, 9, 2]])
    weights = build_token_weights(labels, special_token_ids={9}, upweight=4.0)
    # position 1 is special, position 2 is "the position after"; position 4 is
    # special, position 5 follows it; masked position 3 is zeroed.
    expected = torch.tensor([[1.0, 4.0, 4.0, 0.0, 4.0, 4.0]])
    assert torch.equal(weights, expected)


def test_token_weights_affect_loss() -> None:
    logits_s = torch.randn(1, 3, 8)
    logits_t = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])
    flat = distill_loss(logits_s, logits_t, labels)
    weighted = distill_loss(
        logits_s, logits_t, labels, token_weights=torch.tensor([[1.0, 4.0, 4.0]])
    )
    assert not math.isclose(float(flat["loss"]), float(weighted["loss"]), rel_tol=1e-6)


def test_hidden_loss_zero_for_identity_projection() -> None:
    hidden = [torch.randn(2, 4, 6) for _ in range(3)]
    projs = nn.ModuleList([nn.Linear(6, 6) for _ in range(3)])
    for p in projs:
        with torch.no_grad():
            p.weight.copy_(torch.eye(6))
            p.bias.zero_()
    labels = torch.randint(0, 10, (2, 4))
    logits = torch.randn(2, 4, 10)
    out = distill_loss(
        logits,
        logits,
        labels,
        student_hidden=hidden,
        teacher_hidden=[h.clone() for h in hidden],
        hidden_projs=projs,
        layer_map=[(0, 0), (1, 1), (2, 2)],
    )
    assert float(out["hidden"]) < 1e-10


def test_hidden_loss_skipped_when_states_missing() -> None:
    logits = torch.randn(1, 3, 5)
    labels = torch.randint(0, 5, (1, 3))
    out = distill_loss(logits, logits, labels, student_hidden=None, teacher_hidden=None)
    assert float(out["hidden"]) == 0.0
    assert torch.isfinite(out["loss"])


def test_kl_ce_chunked_matches_dense_reference(monkeypatch) -> None:
    """Position-chunked KL/CE must equal the dense full-tensor computation."""
    import distil_vibevoice.distill.losses as losses_mod

    torch.manual_seed(5)
    T = 2.0
    s = torch.randn(2, 9, 19)
    t = torch.randn(2, 9, 19)
    labels = torch.randint(0, 19, (2, 9))
    labels[0, 3] = -100
    w = torch.rand(2, 9) + 0.5

    # Dense reference computed inline (the pre-chunking formula).
    valid = labels != -100
    weights = w * valid.float()
    log_ps = torch.log_softmax(s / T, dim=-1)
    log_pt = torch.log_softmax(t / T, dim=-1)
    kl_pos = (log_pt.exp() * (log_pt - log_ps)).sum(-1) * (T * T)
    safe = labels.masked_fill(~valid, 0)
    ce_pos = nn.functional.cross_entropy(
        s.reshape(-1, 19), safe.reshape(-1), reduction="none"
    ).view_as(safe)
    denom = weights.sum()
    ref_kl = float((kl_pos * weights).sum() / denom)
    ref_ce = float((ce_pos * weights).sum() / denom)

    monkeypatch.setattr(losses_mod, "KL_CE_CHUNK_POSITIONS", 4)  # 18 pos -> 5 chunks
    out = distill_loss(s, t, labels, token_weights=w, T=T)
    assert math.isclose(float(out["kl"]), ref_kl, rel_tol=1e-5)
    assert math.isclose(float(out["ce"]), ref_ce, rel_tol=1e-5)


def test_chunked_loss_gradients_independent_of_chunk_size(monkeypatch) -> None:
    import distil_vibevoice.distill.losses as losses_mod

    torch.manual_seed(6)
    s = torch.randn(1, 10, 12)
    t = torch.randn(1, 10, 12)
    labels = torch.randint(0, 12, (1, 10))

    def run(chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        monkeypatch.setattr(losses_mod, "KL_CE_CHUNK_POSITIONS", chunk)
        s2 = s.clone().requires_grad_(True)
        out = distill_loss(s2, t, labels, w_kl=0.7, w_ce=0.3, w_hidden=0.0)
        out["loss"].backward()
        assert s2.grad is not None
        return out["loss"].detach(), s2.grad

    loss_one_chunk, grad_one_chunk = run(1024)  # everything in a single chunk
    loss_many, grad_many = run(3)  # several checkpointed chunks
    assert torch.allclose(loss_one_chunk, loss_many, atol=1e-6)
    assert torch.allclose(grad_one_chunk, grad_many, atol=1e-6)


def test_default_layer_map_identity_and_coverage() -> None:
    identity = default_layer_map(28, 28)
    assert identity == [(i, i) for i in range(28)]

    # Student deeper than teacher: every teacher layer covered, monotone,
    # last layers paired.
    m = default_layer_map(28, 36)
    t_idx = [t for t, _ in m]
    s_idx = [s for _, s in m]
    assert s_idx == list(range(36))
    assert t_idx == sorted(t_idx)
    assert set(t_idx) == set(range(28))
    assert m[-1] == (27, 35)

    # Teacher deeper than student (the actual 28 -> pruned case keeps depth,
    # but the mapping must also work for depth reduction).
    m2 = default_layer_map(28, 14)
    assert len(m2) == 14
    assert m2[-1] == (27, 13)
    assert [t for t, _ in m2] == sorted(t for t, _ in m2)


# ----------------------------------------------------------------------
# Trainer smoke test on tiny models
# ----------------------------------------------------------------------
class _TinyLM(nn.Module):
    """Minimal causal-LM stand-in exposing the HF output interface."""

    def __init__(self, vocab: int = 32, hidden: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=layers, hidden_size=hidden, vocab_size=vocab
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        x = self.embed(input_ids)
        states = [x]
        for block in self.blocks:
            x = torch.tanh(block(x))
            states.append(x)
        return SimpleNamespace(
            logits=self.lm_head(x),
            hidden_states=tuple(states) if output_hidden_states else None,
        )


def _make_batches(vocab: int, n: int = 2) -> list[dict]:
    gen = torch.Generator().manual_seed(7)
    batches = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (2, 12), generator=gen)
        batches.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": ids.clone(),
                "token_weights": torch.ones(ids.shape, dtype=torch.float32),
            }
        )
    return batches


def test_trainer_smoke_five_steps() -> None:
    torch.manual_seed(3)
    teacher = _TinyLM()
    student = copy.deepcopy(teacher)
    batches = _make_batches(vocab=32)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "lr": 5e-3,
            "warmup_steps": 1,
            "max_steps": 5,
            "grad_accum": 1,
            "T": 2.0,
            "weights": {"kl": 0.5, "ce": 0.3, "hidden": 0.2},
            "out_dir": tmp,
            "log_every": 1,
            "save_every": 100,
            "teacher_device": "cpu",
            "student_device": "cpu",
            "use_8bit_optim": False,
            "bf16": False,
            "hidden_layer_map": "auto",
            "seed": 0,
        }
        trainer = DistillTrainer(student, teacher, batches, cfg)
        trainer.train()

        assert trainer.step == 5
        losses = [rec["loss"] for rec in trainer.history]
        assert len(losses) == 5
        assert all(math.isfinite(v) for v in losses)
        assert losses[-1] < losses[0] * 1.5 + 1e-9

        ckpts = list(Path(tmp).glob("checkpoint_step*.pt"))
        assert ckpts, "no checkpoint written"
        payload = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        assert payload["step"] == 5
        assert "student" in payload and payload["hidden_projs"] is not None
        assert (Path(tmp) / "metrics.jsonl").exists()


def test_trainer_accepts_flattened_yaml_aliases() -> None:
    """Config keys produced by flattening the YAML sections (scripts 07/09)
    must resolve: w_kl/w_ce/w_hidden, layer_map='default', optimizer_8bit,
    direct_8b_fraction + teacher_8b, min_lr_ratio, betas."""
    teacher = _TinyLM()
    student = copy.deepcopy(teacher)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "out_dir": tmp,
            "lr": 1e-4,
            "warmup_steps": 1,
            "max_steps": 2,
            "T": 2.0,
            "w_kl": 0.6,
            "w_ce": 0.25,
            "w_hidden": 0.15,
            "layer_map": "default",
            "optimizer_8bit": True,  # bnb absent in CI -> AdamW fallback
            "betas": [0.9, 0.95],
            "min_lr_ratio": 0.1,
            "teacher_8b": _TinyLM(),
            "direct_8b_fraction": 0.1,
            "teacher_device": "cpu",
            "student_device": "cpu",
            "bf16": False,
        }
        trainer = DistillTrainer(student, teacher, _make_batches(vocab=32), cfg)
        assert (trainer.w_kl, trainer.w_ce, trainer.w_hidden) == (0.6, 0.25, 0.15)
        assert trainer.layer_map == [(0, 0), (1, 1)]
        assert trainer.direct_teacher is not None
        assert trainer.direct_fraction == 0.1
        assert trainer.optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_direct_teacher_config_path_string_is_not_a_module() -> None:
    """YAML flattening (scripts 07/09) leaves the config PATH string under
    'direct_teacher'; it must not shadow the loaded module under 'teacher_8b'
    and must never be .to()'d as if it were a module."""
    teacher = _TinyLM()
    with tempfile.TemporaryDirectory() as tmp:
        base = {
            "out_dir": tmp,
            "max_steps": 1,
            "warmup_steps": 1,
            "teacher_device": "cpu",
            "student_device": "cpu",
            "bf16": False,
        }
        # Path string + loaded 8B module: the module wins.
        t8 = _TinyLM()
        trainer = DistillTrainer(
            copy.deepcopy(teacher),
            teacher,
            _make_batches(vocab=32),
            {**base, "direct_teacher": "models/teacher", "teacher_8b": t8,
             "direct_8b_fraction": 0.1},
        )
        assert trainer.direct_teacher is t8
        assert trainer.direct_fraction == 0.1

        # Path string only (e.g. --no-direct-8b): no direct teacher, no crash.
        trainer2 = DistillTrainer(
            copy.deepcopy(teacher).requires_grad_(True),
            teacher,
            _make_batches(vocab=32),
            {**base, "direct_teacher": "models/teacher", "direct_8b_fraction": 0.0},
        )
        assert trainer2.direct_teacher is None


class _ExportableTinyLM(_TinyLM):
    """_TinyLM with an HF-style save_pretrained that records its calls."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.saved_to: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(str(path))
        (Path(path) / "config.json").write_text("{}")


def test_train_exports_student_in_hf_format() -> None:
    """train() must leave an HF-loadable export in out_dir (scripts 08/09/10
    consume it with from_pretrained), not only checkpoint_step*.pt payloads."""
    torch.manual_seed(8)
    teacher = _TinyLM()
    student = _ExportableTinyLM()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "out_dir": tmp,
            "lr": 1e-3,
            "warmup_steps": 1,
            "max_steps": 2,
            "save_every": 100,
            "teacher_device": "cpu",
            "student_device": "cpu",
            "bf16": False,
        }
        DistillTrainer(student, teacher, _make_batches(vocab=32), cfg).train()
        assert student.saved_to, "save_pretrained was never called"
        assert student.saved_to[-1] == tmp
        assert (Path(tmp) / "config.json").exists()
        assert list(Path(tmp).glob("checkpoint_step*.pt"))  # .pt payload kept too


def test_trainer_resume_from_checkpoint() -> None:
    torch.manual_seed(4)
    teacher = _TinyLM()
    student = copy.deepcopy(teacher)
    batches = _make_batches(vocab=32)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "lr": 5e-3,
            "warmup_steps": 1,
            "max_steps": 3,
            "out_dir": tmp,
            "log_every": 1,
            "save_every": 100,
            "teacher_device": "cpu",
            "student_device": "cpu",
            "bf16": False,
        }
        DistillTrainer(student, teacher, batches, cfg).train()

        cfg2 = {**cfg, "max_steps": 5, "resume": True}
        # The first trainer froze `teacher` in place; re-enable grads on the
        # copy so the resumed student is trainable like the original one.
        student2 = copy.deepcopy(teacher).requires_grad_(True)
        trainer2 = DistillTrainer(student2, teacher, batches, cfg2)
        assert trainer2.step == 3  # resumed
        trainer2.train()
        assert trainer2.step == 5
