"""Minitron-style width pruning of the VibeVoice-ASR Qwen2 LLM backbone.

Two-step API:
  1. :func:`~distil_vibevoice.pruning.importance.compute_importance` runs a few
     calibration batches through the model and collects activation-magnitude
     importance scores per hidden channel / attention head / FFN channel.
  2. :func:`~distil_vibevoice.pruning.prune.prune_qwen2_width` builds a fresh,
     smaller ``Qwen2ForCausalLM`` by keeping the top-scoring channels/heads and
     copying the sliced weights.
"""

from distil_vibevoice.pruning.importance import ImportanceScores, compute_importance
from distil_vibevoice.pruning.prune import prune_connector, prune_qwen2_width

__all__ = [
    "ImportanceScores",
    "compute_importance",
    "prune_connector",
    "prune_qwen2_width",
]
