"""Minitron-style width pruning of a ``transformers`` Qwen2ForCausalLM.

Given :class:`~distil_vibevoice.pruning.importance.ImportanceScores`, build a
fresh smaller model by keeping the top-scoring residual channels (one keep set
shared by ALL layers, embeddings, norms and lm_head), attention heads (head-
granular, GQA-consistent) and FFN channels (per layer), then copy the sliced
weights. ``head_dim`` is preserved (128 for the real teacher), so rotary
embeddings are unchanged.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from distil_vibevoice.pruning.importance import ImportanceScores


def _topk_sorted(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest scores, sorted ascending (order-preserving)."""
    if k > scores.numel():
        raise ValueError(f"cannot keep {k} of {scores.numel()} channels")
    return torch.topk(scores.float(), k).indices.sort().values


def _head_rows(head_idx: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Expand kept head indices into row indices of a packed [n_heads*head_dim, ...] matrix."""
    return (head_idx[:, None] * head_dim + torch.arange(head_dim)).reshape(-1)


def _select_heads(
    head_scores: torch.Tensor,
    kv_scores: torch.Tensor,
    group_size_old: int,
    target_q_heads: int,
    target_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick kept (q_heads, kv_heads) indices, GQA-consistent.

    Keeps the top ``target_kv_heads`` KV groups, then the top
    ``target_q_heads // target_kv_heads`` Q heads *within each kept group*, so
    the pruned model keeps a uniform group size and ``repeat_kv`` semantics.
    Both index tensors are sorted ascending.
    """
    per_group = target_q_heads // target_kv_heads
    if per_group > group_size_old:
        raise ValueError(
            f"target GQA group size {per_group} exceeds original group size {group_size_old}"
        )
    kv_keep = _topk_sorted(kv_scores, target_kv_heads)
    q_keep: list[int] = []
    for g in kv_keep.tolist():
        group = head_scores[g * group_size_old : (g + 1) * group_size_old]
        local = _topk_sorted(group, per_group)
        q_keep.extend((g * group_size_old + local).tolist())
    return torch.tensor(q_keep, dtype=torch.long), kv_keep


def _copy_(dst: torch.Tensor, src: torch.Tensor, what: str) -> None:
    if dst.shape != src.shape:
        raise RuntimeError(f"shape mismatch copying {what}: {tuple(dst.shape)} vs {tuple(src.shape)}")
    dst.copy_(src.to(dst.dtype))


def prune_qwen2_width(
    model,
    scores: ImportanceScores,
    target_hidden: int,
    target_intermediate: int,
    target_q_heads: int,
    target_kv_heads: int,
    tie_word_embeddings: bool = False,
):
    """Return a NEW width-pruned Qwen2ForCausalLM.

    - ``hidden``: single keep-index set (top ``target_hidden`` of
      ``scores.hidden``) shared across all layers, embeddings, norms, lm_head.
    - attention: head-granular slicing of q/k/v/o projections; ``head_dim`` is
      kept unchanged so RoPE is unaffected.
    - FFN: per-layer top ``target_intermediate`` channels.

    Keeping ALL channels/heads reproduces the original model exactly.

    ``tie_word_embeddings`` (default ``False`` — byte-identical to prior
    behaviour): when ``True``, the produced student ties ``lm_head.weight`` to
    ``model.embed_tokens.weight`` (shared storage) and carries
    ``config.tie_word_embeddings = True`` so ``save_pretrained`` /
    ``from_pretrained`` round-trip as tied. Both matrices are pruned along the
    SAME hidden-column keep-index (``keep_h``), so tying is exact.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = model.config
    n_q: int = cfg.num_attention_heads
    n_kv: int = cfg.num_key_value_heads
    head_dim: int = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    group_size_old = n_q // n_kv

    if target_q_heads % target_kv_heads != 0:
        raise AssertionError(
            f"GQA requires target_q_heads % target_kv_heads == 0, got {target_q_heads} % {target_kv_heads}"
        )
    assert target_hidden <= cfg.hidden_size
    assert target_intermediate <= cfg.intermediate_size
    assert target_q_heads <= n_q and target_kv_heads <= n_kv
    if len(scores.hidden) != cfg.hidden_size:
        raise ValueError("scores.hidden length does not match model hidden_size")

    # Force tying if requested, or preserve the source's tie state otherwise.
    tie = bool(tie_word_embeddings) or bool(cfg.tie_word_embeddings)

    keep_h = _topk_sorted(scores.hidden, target_hidden)

    config_kwargs = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=target_hidden,
        intermediate_size=target_intermediate,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=target_q_heads,
        num_key_value_heads=target_kv_heads,
        head_dim=head_dim,  # keep 128: RoPE depends only on head_dim
        hidden_act=cfg.hidden_act,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=tie,
        initializer_range=cfg.initializer_range,
        use_cache=cfg.use_cache,
        attention_dropout=getattr(cfg, "attention_dropout", 0.0),
    )
    # RoPE config: transformers >=5 nests rope_theta under `rope_parameters`;
    # transformers 4.x exposes flat `rope_theta` / `rope_scaling` attributes.
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if rope_parameters is not None:
        config_kwargs["rope_parameters"] = dict(rope_parameters)
    else:
        config_kwargs["rope_theta"] = getattr(cfg, "rope_theta", 10000.0)
        if getattr(cfg, "rope_scaling", None) is not None:
            config_kwargs["rope_scaling"] = cfg.rope_scaling
    for attr in (
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
        "attention_bias",
    ):
        if hasattr(cfg, attr):
            config_kwargs[attr] = getattr(cfg, attr)
    new_cfg = Qwen2Config(**config_kwargs)

    new_model = Qwen2ForCausalLM(new_cfg)
    param = next(model.parameters())
    new_model = new_model.to(device=param.device, dtype=param.dtype)

    old_layers = model.model.layers
    new_layers = new_model.model.layers

    with torch.no_grad():
        _copy_(
            new_model.model.embed_tokens.weight,
            model.model.embed_tokens.weight[:, keep_h],
            "embed_tokens",
        )
        _copy_(new_model.model.norm.weight, model.model.norm.weight[keep_h], "final norm")

        for i, (old, new) in enumerate(zip(old_layers, new_layers)):
            q_keep, kv_keep = _select_heads(
                scores.heads[i], scores.kv_heads[i], group_size_old, target_q_heads, target_kv_heads
            )
            keep_f = _topk_sorted(scores.ffn[i], target_intermediate)
            q_rows = _head_rows(q_keep, head_dim)
            kv_rows = _head_rows(kv_keep, head_dim)

            oa, na = old.self_attn, new.self_attn
            _copy_(na.q_proj.weight, oa.q_proj.weight[q_rows][:, keep_h], f"layer{i}.q_proj")
            _copy_(na.k_proj.weight, oa.k_proj.weight[kv_rows][:, keep_h], f"layer{i}.k_proj")
            _copy_(na.v_proj.weight, oa.v_proj.weight[kv_rows][:, keep_h], f"layer{i}.v_proj")
            _copy_(na.o_proj.weight, oa.o_proj.weight[keep_h][:, q_rows], f"layer{i}.o_proj")
            for name, rows in (("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)):
                ob = getattr(oa, name).bias
                nb = getattr(na, name).bias
                if ob is not None and nb is not None:
                    _copy_(nb, ob[rows], f"layer{i}.{name}.bias")
            if oa.o_proj.bias is not None and na.o_proj.bias is not None:
                _copy_(na.o_proj.bias, oa.o_proj.bias[keep_h], f"layer{i}.o_proj.bias")

            om, nm = old.mlp, new.mlp
            _copy_(nm.gate_proj.weight, om.gate_proj.weight[keep_f][:, keep_h], f"layer{i}.gate_proj")
            _copy_(nm.up_proj.weight, om.up_proj.weight[keep_f][:, keep_h], f"layer{i}.up_proj")
            _copy_(nm.down_proj.weight, om.down_proj.weight[keep_h][:, keep_f], f"layer{i}.down_proj")

            _copy_(new.input_layernorm.weight, old.input_layernorm.weight[keep_h], f"layer{i}.input_ln")
            _copy_(
                new.post_attention_layernorm.weight,
                old.post_attention_layernorm.weight[keep_h],
                f"layer{i}.post_attn_ln",
            )

        if tie:
            # Tie: lm_head shares embed_tokens storage. embed_tokens was pruned
            # above along keep_h; a tied lm_head uses the SAME keep_h columns, so
            # tying is exact — no column-index mismatch is possible.
            new_model.lm_head.weight = new_model.model.embed_tokens.weight
            new_model.tie_weights()
        else:
            _copy_(new_model.lm_head.weight, model.lm_head.weight[:, keep_h], "lm_head")

    new_model.eval()
    return new_model


def _prune_linear(
    linear: nn.Linear, keep_idx: torch.Tensor, full_dim: int
) -> nn.Linear:
    """Slice a Linear's in/out features wherever they equal ``full_dim``."""
    slice_out = linear.out_features == full_dim
    slice_in = linear.in_features == full_dim
    if not (slice_out or slice_in):
        return linear
    out_f = keep_idx.numel() if slice_out else linear.out_features
    in_f = keep_idx.numel() if slice_in else linear.in_features
    new = nn.Linear(in_f, out_f, bias=linear.bias is not None)
    w = linear.weight
    if slice_out:
        w = w[keep_idx]
    if slice_in:
        w = w[:, keep_idx]
    with torch.no_grad():
        new.weight.copy_(w)
        if linear.bias is not None:
            b = linear.bias[keep_idx] if slice_out else linear.bias
            new.bias.copy_(b)
    return new.to(device=linear.weight.device, dtype=linear.weight.dtype)


def prune_connector(connector: nn.Module, hidden_keep_idx: torch.Tensor) -> nn.Module:
    """Width-prune a connector MLP that projects into the LLM hidden space.

    The original LLM hidden size is inferred from the LAST Linear's
    ``out_features`` (connectors project encoder latents INTO the LLM residual
    stream). Every Linear whose ``in_features`` and/or ``out_features`` equals
    that size is sliced along the matching dimension(s) with
    ``hidden_keep_idx``. Handles a bare ``nn.Linear`` or any container
    (e.g. ``nn.Sequential``) of Linears; other modules pass through untouched.

    Returns a NEW module (deep copy); the input is not modified.
    """
    hidden_keep_idx = hidden_keep_idx.to(torch.long).sort().values
    connector = copy.deepcopy(connector)
    linears = [m for m in connector.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("connector contains no nn.Linear modules")
    full_dim = linears[-1].out_features
    if int(hidden_keep_idx.max()) >= full_dim:
        raise ValueError(
            f"hidden_keep_idx max {int(hidden_keep_idx.max())} out of range for "
            f"inferred hidden size {full_dim}"
        )

    if isinstance(connector, nn.Linear):
        return _prune_linear(connector, hidden_keep_idx, full_dim)

    def _recurse(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, _prune_linear(child, hidden_keep_idx, full_dim))
            else:
                _recurse(child)

    _recurse(connector)
    return connector
