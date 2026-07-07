"""Distillation losses for VibeVoice-ASR width-pruned students.

Combined objective (per the project distillation plan)::

    L = w_kl * KL(teacher || student, T)  +  w_ce * CE(labels)  +  w_hidden * MSE(hidden)

Alignment convention: all tensors passed to :func:`distill_loss` are assumed
to be *already aligned*, i.e. ``logits[:, i]`` predicts ``labels[:, i]``.
Callers that hold HF-style ``labels`` (aligned with ``input_ids``) must apply
the causal shift themselves (``logits[:, :-1]`` vs ``labels[:, 1:]``);
:class:`distil_vibevoice.distill.trainer.DistillTrainer` does exactly that.

Hidden-state convention: ``student_hidden`` / ``teacher_hidden`` are sequences
indexed directly by the ``layer_map`` entries, i.e. entry ``k`` is the output
of decoder layer ``k``. When passing HF ``output.hidden_states`` (which has
the embedding output at index 0), slice off the first element first.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["build_token_weights", "distill_loss", "default_layer_map"]

IGNORE_INDEX = -100

#: Positions per chunk for the KL/CE computation. Full-vocab fp32 temporaries
#: (log_softmax x2, per-position KL) are materialized for at most this many
#: positions at a time, bounding peak loss-side memory to O(chunk * vocab)
#: instead of O(batch * seq_len * vocab) — at the project's 16k-context /
#: 152k-vocab settings the un-chunked version needs tens of GB and OOMs.
KL_CE_CHUNK_POSITIONS = 1024


def build_token_weights(
    labels: "torch.Tensor",
    special_token_ids: set[int],
    upweight: float = 4.0,
) -> "torch.Tensor":
    """Per-position loss weights that upweight speaker-tag / timestamp tokens.

    Returns a float tensor shaped like ``labels`` that is:

    * ``0.0`` where ``labels == -100`` (masked positions contribute nothing),
    * ``upweight`` where the label id is in ``special_token_ids`` **and** at
      the position immediately after such a token (covers multi-token
      speaker/timestamp tags whose continuation pieces are ordinary ids),
    * ``1.0`` everywhere else.
    """
    weights = torch.ones_like(labels, dtype=torch.float32)
    valid = labels != IGNORE_INDEX

    if special_token_ids:
        ids = torch.tensor(
            sorted(special_token_ids), dtype=labels.dtype, device=labels.device
        )
        special = torch.isin(labels, ids) & valid
        # Also upweight the position right after a tag start (multi-token tags).
        after = F.pad(special, (1, 0))[..., :-1]
        weights = torch.where(special | after, torch.full_like(weights, upweight), weights)

    weights = weights * valid.to(weights.dtype)
    return weights


def default_layer_map(teacher_layers: int, student_layers: int) -> list[tuple[int, int]]:
    """Uniform (teacher_layer, student_layer) mapping.

    Student layer ``i`` maps to teacher layer ``round((i + 1) * T / S) - 1``.
    The mapping is monotonically non-decreasing in the teacher index and always
    includes the last layer of both models. For equal depths it is the
    identity ``[(0, 0), (1, 1), ...]``.
    """
    if teacher_layers <= 0 or student_layers <= 0:
        raise ValueError("layer counts must be positive")
    pairs: list[tuple[int, int]] = []
    for i in range(student_layers):
        t = round((i + 1) * teacher_layers / student_layers) - 1
        t = min(max(t, 0), teacher_layers - 1)
        pairs.append((t, i))
    # Guarantee the final layers are paired regardless of rounding mode.
    pairs[-1] = (teacher_layers - 1, student_layers - 1)
    return pairs


def _weighted_mean(per_pos: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
    denom = weights.sum().clamp_min(1e-8)
    return (per_pos * weights).sum() / denom


def _kl_ce_chunk_sums(
    s_chunk: "torch.Tensor",
    t_chunk: "torch.Tensor",
    labels_chunk: "torch.Tensor",
    weights_chunk: "torch.Tensor",
    T: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Weighted KL and CE *sums* over one flat ``[n, vocab]`` position chunk.

    Casts only the chunk to fp32; the full-vocab temporaries live for the
    duration of this call only.
    """
    s = s_chunk.float()
    t = t_chunk.float()
    log_p_student = F.log_softmax(s / T, dim=-1)
    log_p_teacher = F.log_softmax(t / T, dim=-1)
    kl_pos = F.kl_div(
        log_p_student, log_p_teacher, reduction="none", log_target=True
    ).sum(-1) * (T * T)
    ce_pos = F.cross_entropy(s, labels_chunk, reduction="none")
    return (kl_pos * weights_chunk).sum(), (ce_pos * weights_chunk).sum()


def distill_loss(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    labels: "torch.Tensor",
    token_weights: "torch.Tensor | None" = None,
    T: float = 2.0,
    w_kl: float = 0.5,
    w_ce: float = 0.3,
    w_hidden: float = 0.2,
    student_hidden: "Sequence[torch.Tensor] | None" = None,
    teacher_hidden: "Sequence[torch.Tensor] | None" = None,
    hidden_projs: "torch.nn.ModuleList | None" = None,
    layer_map: list[tuple[int, int]] | None = None,
    kept_vocab_ids: "torch.Tensor | None" = None,
    label_remap: "torch.Tensor | None" = None,
) -> dict:
    """Combined KD loss. Returns ``{'loss', 'kl', 'ce', 'hidden'}`` (tensors).

    * ``kl``: temperature-scaled KL(teacher || student), summed over the vocab,
      scaled by ``T**2``, then weight-averaged over valid (label != -100)
      positions using ``token_weights`` (ones if None).
    * ``ce``: per-position cross entropy against ``labels``, same weighting.
    * ``hidden``: mean over ``layer_map`` pairs ``(t_layer, s_layer)`` of the
      MSE between ``student_hidden[s_layer]`` and ``hidden_projs[k](teacher_hidden[t_layer])``
      (projection applied to the *teacher* states), averaged over valid
      positions. Skipped cleanly (0.0) when any hidden argument is None or
      ``w_hidden <= 0``.

    Trimmed-vocab distillation (``kept_vocab_ids``)
    ----------------------------------------------
    When the student keeps only a subset ``K = {k_0, ..., k_{m-1}}`` of the
    teacher's ``V`` vocabulary ids, pass ``kept_vocab_ids`` as a 1-D ``long``
    tensor of those ids (the order defines the student's column order, so
    student column ``j`` corresponds to teacher id ``kept_vocab_ids[j]``).
    ``teacher_logits`` must still be **full-vocab** ``[..., V]`` while
    ``student_logits`` is ``[..., m]``. The teacher columns are gathered to the
    kept set and the distribution is renormalized *implicitly* by the
    ``log_softmax`` over the gathered subset::

        p_T^{kept}(k_j) = exp(z_{k_j} / T) / sum_{k in K} exp(z_k / T)
                        = p_T(k_j) / sum_{k in K} p_T(k)

    i.e. the probability mass on dropped ids is divided out. This is the
    mathematically-correct target for distilling a full-vocab teacher into a
    trimmed-vocab student: the KL is taken over the kept set only.

    Hard ``labels`` are remapped from teacher-vocab ids to student column
    indices: ids not in ``K`` become ``ignore_index`` (-100) for the CE term,
    ids in ``K`` become their new column position. Provide ``label_remap`` (a
    ``[V]`` long tensor mapping id -> new column or -100) to reuse a precomputed
    lookup; otherwise it is derived from ``kept_vocab_ids``. Behavior is
    identical to the dense path when ``kept_vocab_ids is None`` or when it is
    ``arange(V)`` (the identity kept set).
    """
    labels = labels.long()
    if kept_vocab_ids is not None:
        kept_vocab_ids = kept_vocab_ids.to(device=teacher_logits.device, dtype=torch.long)
        full_vocab = teacher_logits.size(-1)
        # Restrict the teacher distribution to the columns the student keeps.
        # A log_softmax over this gathered subset *is* the renormalized teacher
        # distribution p_T(y | y in kept): the dropped-token mass is divided
        # out, giving the correct target for a trimmed-vocab student.
        teacher_logits = teacher_logits.index_select(-1, kept_vocab_ids)
        if label_remap is None:
            label_remap = torch.full(
                (full_vocab,), IGNORE_INDEX, dtype=torch.long, device=labels.device
            )
            label_remap[kept_vocab_ids.to(labels.device)] = torch.arange(
                kept_vocab_ids.numel(), device=labels.device
            )
        else:
            label_remap = label_remap.to(device=labels.device, dtype=torch.long)
        keep_label = labels != IGNORE_INDEX
        # clamp_min(0) keeps the gather in-bounds for already-ignored (-100)
        # positions; torch.where restores their -100 afterwards.
        labels = torch.where(keep_label, label_remap[labels.clamp_min(0)], labels)
    valid = labels != IGNORE_INDEX
    if token_weights is None:
        weights = valid.to(torch.float32)
    else:
        weights = token_weights.to(torch.float32) * valid.to(torch.float32)

    # --- KL(teacher || student) with temperature + CE against hard labels ---
    # Computed in position chunks so at most KL_CE_CHUNK_POSITIONS x vocab
    # fp32 temporaries exist at once (exact same result as the dense
    # computation: the weighted mean decomposes into chunked weighted sums).
    # Chunks are gradient-checkpointed so backward, too, materializes one
    # chunk's fp32 tensors at a time (only the input logit slices are saved).
    vocab = student_logits.size(-1)
    flat_s = student_logits.reshape(-1, vocab)
    flat_t = teacher_logits.reshape(-1, vocab)
    safe_labels = labels.masked_fill(~valid, 0)
    flat_labels = safe_labels.reshape(-1)
    flat_weights = weights.reshape(-1)
    denom = weights.sum().clamp_min(1e-8)

    chunk = max(int(KL_CE_CHUNK_POSITIONS), 1)
    use_ckpt = torch.is_grad_enabled() and (
        flat_s.requires_grad or flat_t.requires_grad
    )
    kl_sum = student_logits.new_zeros((), dtype=torch.float32)
    ce_sum = student_logits.new_zeros((), dtype=torch.float32)
    for lo in range(0, flat_s.size(0), chunk):
        hi = lo + chunk
        args = (flat_s[lo:hi], flat_t[lo:hi], flat_labels[lo:hi], flat_weights[lo:hi], T)
        if use_ckpt:
            kl_c, ce_c = checkpoint(_kl_ce_chunk_sums, *args, use_reentrant=False)
        else:
            kl_c, ce_c = _kl_ce_chunk_sums(*args)
        kl_sum = kl_sum + kl_c
        ce_sum = ce_sum + ce_c
    kl = kl_sum / denom
    ce = ce_sum / denom

    # --- Hidden-state MSE over mapped layers ---
    zero = student_logits.new_zeros((), dtype=torch.float32)
    hidden = zero
    if w_hidden > 0 and student_hidden is not None and teacher_hidden is not None:
        if layer_map is None:
            layer_map = default_layer_map(len(teacher_hidden), len(student_hidden))
        pos_weights = valid.to(torch.float32)
        terms = []
        for k, (t_layer, s_layer) in enumerate(layer_map):
            t_state = teacher_hidden[t_layer].float()
            if hidden_projs is not None:
                t_state = hidden_projs[k](t_state)
            diff = (student_hidden[s_layer].float() - t_state).pow(2).mean(-1)
            terms.append(_weighted_mean(diff, pos_weights))
        if terms:
            hidden = torch.stack(terms).mean()

    loss = w_kl * kl + w_ce * ce + w_hidden * hidden
    return {"loss": loss, "kl": kl, "ce": ce, "hidden": hidden}
