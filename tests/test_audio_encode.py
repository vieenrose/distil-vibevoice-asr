"""Grad-flow contract for distill.audio_encode (CPU, tiny mock of VibeVoice-ASR)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from distil_vibevoice.distill.audio_encode import enable_connector_training


class _EncOut:
    def __init__(self, t):
        self._t = t
        self.mean = t

    def sample(self, dist_type=None):
        return [self._t]


class _Encoder(torch.nn.Module):
    """Stand-in σ-VAE tokenizer: a conv that must stay frozen."""
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(1, dim)
        self.std_dist_type = "fix"

    def encode(self, x):  # x: (B,1,T) -> tokens (B,T,dim)
        t = x.squeeze(1).unsqueeze(-1)  # (B,T,1)
        return _EncOut(self.proj(t))


class _Inner(torch.nn.Module):
    def __init__(self, vae_dim, hidden):
        super().__init__()
        self.acoustic_tokenizer = _Encoder(vae_dim)
        self.semantic_tokenizer = _Encoder(vae_dim)
        self.acoustic_connector = torch.nn.Linear(vae_dim, hidden)
        self.semantic_connector = torch.nn.Linear(vae_dim, hidden)


class _Model(torch.nn.Module):
    def __init__(self, vae_dim=4, hidden=8):
        super().__init__()
        self.model = _Inner(vae_dim, hidden)

        class _Cfg:
            torch_dtype = torch.float32

        self.config = _Cfg()


def test_connectors_get_grad_encoders_do_not():
    m = _Model()
    conn_params = enable_connector_training(m)
    assert len(conn_params) == 4  # 2 connectors x (weight+bias)

    speech = torch.randn(1, 240)  # ~short clip
    feats = m.encode_speech(speech)
    feats.sum().backward()

    # connectors received gradient
    assert m.model.acoustic_connector.weight.grad is not None
    assert m.model.semantic_connector.weight.grad is not None
    # frozen encoders received none (requires_grad False -> grad stays None)
    assert m.model.acoustic_tokenizer.proj.weight.grad is None
    assert not m.model.acoustic_tokenizer.proj.weight.requires_grad


def test_long_audio_rejected():
    m = _Model()
    enable_connector_training(m)
    long = torch.randn(1, 24000 * 61)  # > 60 s
    with pytest.raises(ValueError, match="non-streaming"):
        m.encode_speech(long)
