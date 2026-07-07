"""distil-vibevoice-asr: prune-and-distill VibeVoice-ASR for on-device meeting transcription.

Cascade: microsoft/VibeVoice-ASR (8.7B, Qwen2.5-7B backbone) -> 4B -> 1.5B (int4 QAT),
targeting zh-TW + English code-switched meetings with speaker diarization and timestamps
on a 6 GB-RAM phone.

Subpackages
-----------
- ``distil_vibevoice.data``: manifests, zh-TW normalization, pseudo-labeling, synthesis,
  augmentation, meeting simulation, dedupe.
- ``distil_vibevoice.pruning``: Minitron-style width importance scoring and pruning.
- ``distil_vibevoice.distill``: losses, collator, two-GPU distillation trainer.
- ``distil_vibevoice.eval``: MER / cpWER / DER / timestamp MAE and release gates.
- ``distil_vibevoice.runtime``: chunked long-form inference, speaker stitching, RAM budget.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
