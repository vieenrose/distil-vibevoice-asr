"""Plain-PyTorch knowledge-distillation trainer.

Single-node, two-GPU friendly loop (teacher on ``cuda:1``, student on
``cuda:0`` by default) with bf16 autocast, gradient accumulation, cosine LR
schedule with warmup, gradient clipping, checkpointing/resume, JSONL metrics
logging, an optional 8-bit optimizer, and an optional *secondary* teacher
applied on a random fraction of steps (the stage-2 "distill 10% of batches
directly from the 8B" skip connection).

Config keys (all optional except where noted) mirror
``configs/distill_stage1_4b.yaml``::

    lr, warmup_steps, max_steps, grad_accum, T,
    weights: {kl, ce, hidden},          # or flat w_kl / w_ce / w_hidden
    out_dir (required), log_every, save_every,
    teacher_device, student_device, bf16,
    use_8bit_optim (alias: optimizer_8bit),
    hidden_layer_map ('auto' or [[t, s], ...]; alias: layer_map, 'default'),
    direct_teacher (alias: teacher_8b; must be an nn.Module — non-module
    values such as the YAML path string are ignored),
    direct_teacher_fraction (alias: direct_8b_fraction), direct_teacher_device,
    min_lr_ratio, betas, grad_clip, weight_decay,
    resume (bool) / resume_from (path), seed, forward_keys

Aliases exist so a config produced by flattening the ``loss``/``optim``/
``train`` sections of the YAML files (as ``scripts/07_distill_4b.py`` and
``scripts/09_distill_1p5b.py`` do) works unchanged.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from distil_vibevoice.distill.losses import default_layer_map, distill_loss

__all__ = ["DistillTrainer"]

_DEFAULTS: dict[str, Any] = {
    "lr": 2e-4,
    "warmup_steps": 100,
    "max_steps": 10000,
    "grad_accum": 1,
    "T": 2.0,
    "weights": {"kl": 0.5, "ce": 0.3, "hidden": 0.2},
    "log_every": 10,
    "save_every": 500,
    "teacher_device": "cuda:1",
    "student_device": "cuda:0",
    "use_8bit_optim": False,
    "bf16": True,
    "hidden_layer_map": "auto",
    "direct_teacher_fraction": 0.0,
    "seed": 0,
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "forward_keys": ["input_ids", "attention_mask"],
}


def _num_layers(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise ValueError("model.config.num_hidden_layers is required for hidden distillation")
    return int(n)


def _hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    h = getattr(cfg, "hidden_size", None)
    if h is None:
        raise ValueError("model.config.hidden_size is required for hidden distillation")
    return int(h)


def _cycle(loader: Iterable) -> Iterable:
    while True:
        yield from loader


class DistillTrainer:
    """Distill ``teacher`` into ``student`` over ``train_loader`` batches.

    Batches are dicts produced by
    :class:`distil_vibevoice.distill.collator.DistillCollator` (``input_ids``,
    ``attention_mask``, ``labels``, ``token_weights``, ...). The trainer
    applies the causal shift before calling
    :func:`distil_vibevoice.distill.losses.distill_loss`.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: Iterable,
        cfg: dict,
    ) -> None:
        self.cfg = {**_DEFAULTS, **cfg}
        if "out_dir" not in self.cfg:
            raise ValueError("cfg['out_dir'] is required")
        self.out_dir = Path(self.cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights: flat w_kl/w_ce/w_hidden (flattened-YAML style) beat the
        # nested `weights` dict, which beats the defaults.
        w = dict(_DEFAULTS["weights"])
        if isinstance(cfg.get("weights"), dict):
            nested = cfg["weights"]
            for key, alias in (("kl", "w_kl"), ("ce", "w_ce"), ("hidden", "w_hidden")):
                if alias in nested:
                    w[key] = nested[alias]
                if key in nested:
                    w[key] = nested[key]
        self.w_kl = float(cfg.get("w_kl", w["kl"]))
        self.w_ce = float(cfg.get("w_ce", w["ce"]))
        self.w_hidden = float(cfg.get("w_hidden", w["hidden"]))
        self.T = float(self.cfg["T"])

        self.student_device = torch.device(self.cfg["student_device"])
        self.teacher_device = torch.device(self.cfg["teacher_device"])

        self.student = student.to(self.student_device)
        self.student.train()
        self.teacher = teacher.to(self.teacher_device)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        # Optional secondary teacher (stage-2 direct 8B distillation).
        # Only nn.Module values count: YAML configs carry a *path string*
        # under 'direct_teacher' (e.g. 'models/teacher'), which must not
        # shadow the loaded module the launcher wires in under 'teacher_8b'.
        dt = self.cfg.get("direct_teacher")
        if not isinstance(dt, nn.Module):
            dt = self.cfg.get("teacher_8b")
        self.direct_teacher: nn.Module | None = dt if isinstance(dt, nn.Module) else None
        self.direct_fraction = float(
            cfg.get(
                "direct_teacher_fraction",
                cfg.get("direct_8b_fraction", _DEFAULTS["direct_teacher_fraction"]),
            )
        )
        if self.direct_teacher is not None:
            dt_device = torch.device(
                self.cfg.get("direct_teacher_device", self.cfg["teacher_device"])
            )
            self.direct_teacher = self.direct_teacher.to(dt_device)
            self.direct_teacher.eval()
            self.direct_teacher.requires_grad_(False)
            self._direct_device = dt_device

        self.train_loader = train_loader
        self._rng = random.Random(self.cfg["seed"])

        # Hidden-state projections (teacher_hidden -> student_hidden per pair).
        self.layer_map: list[tuple[int, int]] | None = None
        self.hidden_projs: nn.ModuleList | None = None
        if self.w_hidden > 0:
            lm = cfg.get("hidden_layer_map", cfg.get("layer_map", "auto"))
            if lm in ("auto", "default") or lm is None:
                self.layer_map = default_layer_map(
                    _num_layers(self.teacher), _num_layers(self.student)
                )
            else:
                self.layer_map = [tuple(p) for p in lm]  # type: ignore[misc]
            t_h, s_h = _hidden_size(self.teacher), _hidden_size(self.student)
            self.hidden_projs = nn.ModuleList(
                [nn.Linear(t_h, s_h) for _ in self.layer_map]
            ).to(self.student_device)

        self._trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        if self.hidden_projs is not None:
            self._trainable_params += list(self.hidden_projs.parameters())

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.step = 0
        self.history: list[dict] = []
        self._metrics_path = self.out_dir / "metrics.jsonl"

        resume_from = self.cfg.get("resume_from")
        if resume_from is None and self.cfg.get("resume"):
            resume_from = self._latest_checkpoint()
        if resume_from:
            self.load_checkpoint(resume_from)

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.cfg["lr"])
        wd = float(self.cfg["weight_decay"])
        betas = tuple(float(b) for b in self.cfg.get("betas", (0.9, 0.999)))
        use_8bit = bool(self.cfg.get("use_8bit_optim") or self.cfg.get("optimizer_8bit"))
        if use_8bit:
            try:
                import bitsandbytes as bnb

                return bnb.optim.AdamW8bit(
                    self._trainable_params, lr=lr, weight_decay=wd, betas=betas
                )
            except ImportError:
                print("[DistillTrainer] bitsandbytes not installed; falling back to AdamW")
        return torch.optim.AdamW(
            self._trainable_params, lr=lr, weight_decay=wd, betas=betas
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        warmup = max(int(self.cfg["warmup_steps"]), 0)
        max_steps = max(int(self.cfg["max_steps"]), 1)
        min_ratio = min(max(float(self.cfg.get("min_lr_ratio", 0.0)), 0.0), 1.0)

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(max_steps - warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    def _latest_checkpoint(self) -> str | None:
        best: tuple[int, Path] | None = None
        for p in self.out_dir.glob("checkpoint_step*.pt"):
            m = re.search(r"checkpoint_step(\d+)\.pt$", p.name)
            if m:
                s = int(m.group(1))
                if best is None or s > best[0]:
                    best = (s, p)
        return str(best[1]) if best else None

    def save_checkpoint(self) -> Path:
        path = self.out_dir / f"checkpoint_step{self.step}.pt"
        payload: dict[str, Any] = {
            "step": self.step,
            "student": self.student.state_dict(),
            "hidden_projs": self.hidden_projs.state_dict()
            if self.hidden_projs is not None
            else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(payload, path)
        return path

    def export_hf(self) -> None:
        """Export the student (and tokenizer, if provided) in HF format.

        Downstream pipeline stages load ``out_dir`` with
        ``Qwen2ForCausalLM.from_pretrained`` (scripts 08/09/10), which cannot
        read the ``checkpoint_step*.pt`` payloads — so the finished student
        must also be written with ``save_pretrained``.
        """
        save = getattr(self.student, "save_pretrained", None)
        if callable(save):
            save(str(self.out_dir))
        else:
            print(
                "[DistillTrainer] student has no save_pretrained(); "
                f"skipping HF export to {self.out_dir}"
            )
        tokenizer = self.cfg.get("tokenizer")
        tok_save = getattr(tokenizer, "save_pretrained", None)
        if callable(tok_save):
            tok_save(str(self.out_dir))

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.student_device, weights_only=False)
        self.student.load_state_dict(payload["student"])
        if self.hidden_projs is not None and payload.get("hidden_projs") is not None:
            self.hidden_projs.load_state_dict(payload["hidden_projs"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.step = int(payload["step"])
        print(f"[DistillTrainer] resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------
    def _forward(
        self, model: nn.Module, batch: dict, device: torch.device, need_hidden: bool
    ):
        kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k in self.cfg["forward_keys"]
        }
        return model(**kwargs, output_hidden_states=need_hidden)

    def _micro_step(self, batch: dict) -> dict:
        """Forward student + teacher on one micro-batch and return loss dict."""
        use_direct = (
            self.direct_teacher is not None
            and self._rng.random() < self.direct_fraction
        )
        # Hidden distillation only against the primary teacher (the projection
        # dims are built for it); direct-teacher steps use logit KL + CE only.
        want_hidden = self.w_hidden > 0 and not use_direct

        student_out = self._forward(
            self.student, batch, self.student_device, need_hidden=want_hidden
        )
        with torch.no_grad():
            if use_direct:
                teacher_out = self._forward(
                    self.direct_teacher, batch, self._direct_device, need_hidden=False
                )
            else:
                teacher_out = self._forward(
                    self.teacher, batch, self.teacher_device, need_hidden=want_hidden
                )

        labels = batch["labels"].to(self.student_device)
        weights = batch.get("token_weights")
        if weights is not None:
            weights = weights.to(self.student_device)

        # Causal shift: logits at position i predict token i+1.
        s_logits = student_out.logits[:, :-1]
        t_logits = teacher_out.logits[:, :-1].to(self.student_device)
        labels_s = labels[:, 1:]
        weights_s = weights[:, 1:] if weights is not None else None

        s_hidden = t_hidden = None
        if want_hidden:
            # Drop the embedding output (index 0) so entries index decoder layers,
            # and trim the last position to match the shifted logits/labels.
            s_hidden = [h[:, :-1] for h in student_out.hidden_states[1:]]
            t_hidden = [
                h[:, :-1].to(self.student_device) for h in teacher_out.hidden_states[1:]
            ]

        return distill_loss(
            s_logits,
            t_logits,
            labels_s,
            token_weights=weights_s,
            T=self.T,
            w_kl=self.w_kl,
            w_ce=self.w_ce,
            w_hidden=self.w_hidden if want_hidden else 0.0,
            student_hidden=s_hidden,
            teacher_hidden=t_hidden,
            hidden_projs=self.hidden_projs,
            layer_map=self.layer_map,
        )

    # ------------------------------------------------------------------
    def _log(self, record: dict) -> None:
        self.history.append(record)
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def train(self) -> None:
        """Run the distillation loop for ``cfg['max_steps']`` optimizer steps."""
        max_steps = int(self.cfg["max_steps"])
        grad_accum = max(int(self.cfg["grad_accum"]), 1)
        log_every = max(int(self.cfg["log_every"]), 1)
        save_every = max(int(self.cfg["save_every"]), 1)
        clip = float(self.cfg["grad_clip"])
        autocast_enabled = bool(self.cfg["bf16"])
        device_type = self.student_device.type

        progress = None
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=max_steps, initial=self.step, desc="distill")
        except ImportError:
            pass

        batches = iter(_cycle(self.train_loader))
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        while self.step < max_steps:
            accum: dict[str, float] = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "hidden": 0.0}
            for _ in range(grad_accum):
                batch = next(batches)
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    out = self._micro_step(batch)
                (out["loss"] / grad_accum).backward()
                for k in accum:
                    accum[k] += float(out[k].detach()) / grad_accum

            torch.nn.utils.clip_grad_norm_(self._trainable_params, clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if progress is not None:
                progress.update(1)
                progress.set_postfix(loss=f"{accum['loss']:.4f}")

            if self.step % log_every == 0 or self.step == 1:
                record = {
                    "step": self.step,
                    "lr": self.scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - t0, 2),
                    **{k: round(v, 6) for k, v in accum.items()},
                }
                self._log(record)
                if progress is None:
                    print(f"[DistillTrainer] {record}")

            if self.step % save_every == 0:
                self.save_checkpoint()

        if progress is not None:
            progress.close()
        self.save_checkpoint()
        self.export_hf()
