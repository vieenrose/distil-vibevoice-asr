"""CPU tests for Minitron-style width pruning (importance + prune).

Uses a tiny randomly-initialized Qwen2 model; no GPU, no network.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="transformers required for pruning tests")

from distil_vibevoice.pruning import (  # noqa: E402
    ImportanceScores,
    compute_importance,
    prune_connector,
    prune_qwen2_width,
)

# Must match the tiny_qwen2 fixture in tests/conftest.py.
VOCAB = 512
HIDDEN = 64
INTERMEDIATE = 128
LAYERS = 2
Q_HEADS = 4
KV_HEADS = 2  # head_dim implicit: 64 / 4 = 16


@pytest.fixture()
def calib_loader():
    torch.manual_seed(1)
    return [{"input_ids": torch.randint(0, VOCAB, (2, 16))} for _ in range(4)]


@pytest.fixture()
def scores(tiny_qwen2, calib_loader) -> ImportanceScores:
    return compute_importance(tiny_qwen2, calib_loader, num_batches=4, device="cpu")


def test_importance_shapes(scores: ImportanceScores) -> None:
    assert scores.hidden.shape == (HIDDEN,)
    assert len(scores.heads) == LAYERS and len(scores.kv_heads) == LAYERS
    assert len(scores.ffn) == LAYERS
    for layer in range(LAYERS):
        assert scores.heads[layer].shape == (Q_HEADS,)
        assert scores.kv_heads[layer].shape == (KV_HEADS,)
        assert scores.ffn[layer].shape == (INTERMEDIATE,)
    for t in [scores.hidden, *scores.heads, *scores.kv_heads, *scores.ffn]:
        assert torch.isfinite(t).all()
        assert (t >= 0).all()


def test_prune_shapes_and_forward(tiny_qwen2, scores) -> None:
    pruned = prune_qwen2_width(
        tiny_qwen2,
        scores,
        target_hidden=48,
        target_intermediate=96,
        target_q_heads=2,
        target_kv_heads=2,
    )
    cfg = pruned.config
    assert cfg.hidden_size == 48
    assert cfg.intermediate_size == 96
    assert cfg.num_attention_heads == 2
    assert cfg.num_key_value_heads == 2
    # head_dim preserved (RoPE unchanged), not re-derived from hidden/heads.
    assert (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads) == 16
    assert cfg.vocab_size == VOCAB

    n_before = sum(p.numel() for p in tiny_qwen2.parameters())
    n_after = sum(p.numel() for p in pruned.parameters())
    assert n_after < n_before

    input_ids = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        out = pruned(input_ids=input_ids)
    assert out.logits.shape == (2, 12, VOCAB)
    assert torch.isfinite(out.logits).all()


def test_identity_pruning_reproduces_logits(tiny_qwen2, scores) -> None:
    """Keeping ALL channels/heads must reproduce the original model exactly."""
    pruned = prune_qwen2_width(
        tiny_qwen2,
        scores,
        target_hidden=HIDDEN,
        target_intermediate=INTERMEDIATE,
        target_q_heads=Q_HEADS,
        target_kv_heads=KV_HEADS,
    )
    input_ids = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        ref = tiny_qwen2(input_ids=input_ids).logits
        hyp = pruned(input_ids=input_ids).logits
    assert torch.allclose(ref, hyp, atol=1e-4), (ref - hyp).abs().max().item()


def test_tied_embeddings_share_storage(tiny_qwen2, scores) -> None:
    """tie_word_embeddings=True: lm_head and embed_tokens are the SAME tensor."""
    tied = prune_qwen2_width(
        tiny_qwen2,
        scores,
        target_hidden=48,
        target_intermediate=96,
        target_q_heads=2,
        target_kv_heads=2,
        tie_word_embeddings=True,
    )
    assert tied.config.tie_word_embeddings is True
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    # Forward still finite.
    input_ids = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        out = tied(input_ids=input_ids)
    assert out.logits.shape == (2, 12, VOCAB)
    assert torch.isfinite(out.logits).all()


def test_tied_param_count_lower_by_vocab_hidden(tiny_qwen2, scores) -> None:
    """Tying removes exactly vocab*hidden params vs the untied prune."""
    kwargs = dict(
        target_hidden=48,
        target_intermediate=96,
        target_q_heads=2,
        target_kv_heads=2,
    )
    untied = prune_qwen2_width(tiny_qwen2, scores, tie_word_embeddings=False, **kwargs)
    tied = prune_qwen2_width(tiny_qwen2, scores, tie_word_embeddings=True, **kwargs)
    assert untied.config.tie_word_embeddings is False
    n_untied = sum(p.numel() for p in untied.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_untied - n_tied == VOCAB * 48


def test_tied_roundtrip_preserves_tying(tiny_qwen2, scores, tmp_path) -> None:
    """save_pretrained -> from_pretrained keeps lm_head tied to embed_tokens."""
    from transformers import Qwen2ForCausalLM

    tied = prune_qwen2_width(
        tiny_qwen2,
        scores,
        target_hidden=48,
        target_intermediate=96,
        target_q_heads=2,
        target_kv_heads=2,
        tie_word_embeddings=True,
    )
    save_dir = tmp_path / "tied_student"
    tied.save_pretrained(save_dir)
    reloaded = Qwen2ForCausalLM.from_pretrained(save_dir)
    assert reloaded.config.tie_word_embeddings is True
    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
    input_ids = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        assert torch.isfinite(reloaded(input_ids=input_ids).logits).all()


def test_untied_default_is_byte_identical(tiny_qwen2, scores) -> None:
    """Default (tie_word_embeddings omitted) keeps separate lm_head storage."""
    pruned = prune_qwen2_width(
        tiny_qwen2,
        scores,
        target_hidden=48,
        target_intermediate=96,
        target_q_heads=2,
        target_kv_heads=2,
    )
    assert pruned.config.tie_word_embeddings is False
    assert pruned.lm_head.weight is not pruned.model.embed_tokens.weight


def test_gqa_divisibility_enforced(tiny_qwen2, scores) -> None:
    with pytest.raises(AssertionError):
        prune_qwen2_width(
            tiny_qwen2,
            scores,
            target_hidden=48,
            target_intermediate=96,
            target_q_heads=3,
            target_kv_heads=2,
        )


def test_prune_connector_linear() -> None:
    torch.manual_seed(2)
    connector = torch.nn.Linear(32, HIDDEN)
    keep = torch.topk(torch.rand(HIDDEN), 48).indices.sort().values
    pruned = prune_connector(connector, keep)
    assert isinstance(pruned, torch.nn.Linear)
    assert pruned.in_features == 32 and pruned.out_features == 48
    x = torch.randn(3, 32)
    with torch.no_grad():
        assert torch.allclose(pruned(x), connector(x)[:, keep], atol=1e-6)
    # Input untouched.
    assert connector.out_features == HIDDEN


def test_prune_connector_sequential() -> None:
    torch.manual_seed(3)
    connector = torch.nn.Sequential(
        torch.nn.Linear(32, HIDDEN),
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
    )
    keep = torch.topk(torch.rand(HIDDEN), 48).indices.sort().values
    pruned = prune_connector(connector, keep)
    assert pruned[0].out_features == 48
    assert pruned[2].in_features == 48 and pruned[2].out_features == 48
    x = torch.randn(3, 32)
    with torch.no_grad():
        y = pruned(x)
    assert y.shape == (3, 48) and torch.isfinite(y).all()

    # Identity property: keeping every channel reproduces the original output.
    all_idx = torch.arange(HIDDEN)
    same = prune_connector(connector, all_idx)
    with torch.no_grad():
        assert torch.allclose(same(x), connector(x), atol=1e-6)
