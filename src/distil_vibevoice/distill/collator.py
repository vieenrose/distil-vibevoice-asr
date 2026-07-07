"""Batch collation for distillation training.

Pads ``input_ids`` / ``labels`` / ``attention_mask``, left-truncates overly
long sequences keeping the *tail* (the transcript target sits at the end of
the sequence), builds per-position ``token_weights`` that upweight
speaker/timestamp markup tokens, and passes through padded ``audio_latents``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import torch

from distil_vibevoice.distill.losses import IGNORE_INDEX, build_token_weights

__all__ = ["DistillCollator"]

# Fallback probe of the teacher's structured output format (VibeVoice-ASR
# generates a JSON-array string with Start/End/Speaker/Content keys), used
# when distil_vibevoice.data.manifest is unavailable at import time.
_FALLBACK_PROBE = '[{"Start":0.0,"End":1.5,"Speaker":0,"Content":"x"}]'

_KEY_PATTERN = re.compile(r'"[A-Za-z_ ]+"\s*:')


def _markup_probe() -> str:
    """Render a probe target string via the canonical formatter if available."""
    try:
        from distil_vibevoice.data.manifest import Segment, format_target

        return format_target(
            [Segment(start=0.0, end=1.5, speaker="0", text="probe text")]
        )
    except Exception:
        return _FALLBACK_PROBE


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Tokenize ``text`` to ids without special tokens, tolerant of tokenizer APIs."""
    try:
        out = tokenizer(text, add_special_tokens=False)
        ids = out["input_ids"] if isinstance(out, dict) or hasattr(out, "__getitem__") else out
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):  # batched output
            ids = ids[0]
        return list(ids)
    except TypeError:
        return list(tokenizer.encode(text))


class DistillCollator:
    """Collate tokenized features into padded distillation batches.

    Each feature is a dict with ``input_ids`` and ``labels`` (lists or 1-D
    tensors of equal length) and optionally ``attention_mask`` and
    ``audio_latents`` (a ``[time, dim]`` float tensor).

    The returned batch contains ``input_ids``, ``attention_mask``, ``labels``,
    ``token_weights`` and, when any feature carries them, ``audio_latents``
    (``[B, T_max, D]``) plus ``audio_latents_mask`` (``[B, T_max]``).
    """

    def __init__(
        self,
        tokenizer: Any,
        max_len: int = 16384,
        speaker_ts_upweight: float = 4.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.speaker_ts_upweight = float(speaker_ts_upweight)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_id: int = int(pad_id) if pad_id is not None else 0

        self.special_token_ids: set[int] = self._collect_special_ids()

    # ------------------------------------------------------------------
    def _collect_special_ids(self) -> set[int]:
        """Ids of tokens forming the speaker/timestamp markup delimiters.

        Computed once by tokenizing the markup delimiters extracted from the
        canonical training-target format (``manifest.format_target``); each
        delimiter is tokenized standalone and with common preceding context so
        that context-sensitive BPE merges are covered.

        With the real Qwen2.5 (152k-vocab BPE) tokenizer the structural glue of
        the JSON-array target merges into multi-char tokens whose id depends on
        the *surrounding* punctuation, e.g. ``{"`` (object open), ``,"`` (field
        separator), ``":`` (key/value colon) and — critically — ``"},{"`` (the
        inter-segment / speaker-turn boundary that opens the next object). The
        open-only context prefixes (``{`` / ``,`` / ``,{``) miss the object
        *close*+separator merge, so those boundary tokens must be seeded from
        close contexts (``"},{`` prefix) and from standalone structural glue
        (``"},{"`` separator, ``"}]`` array terminator, ``[{"`` array opener)
        drawn from the probe itself.
        """
        probe = _markup_probe()
        delimiters = list(dict.fromkeys(_KEY_PATTERN.findall(probe)))
        if not delimiters:  # unexpected format: fall back to known keys
            delimiters = ['"Start":', '"End":', '"Speaker":', '"Content":']

        ids: set[int] = set()
        # Per-key delimiter in open AND close/separator contexts so the BPE
        # merges for both the first object and every subsequent one are covered.
        for delim in delimiters:
            for variant in (
                delim,
                "{" + delim,
                "," + delim,
                ",{" + delim,
                '"},{' + delim,  # object close + separator + next object open
                '"}],{' + delim,
            ):
                try:
                    ids.update(_encode(self.tokenizer, variant))
                except Exception:
                    continue
        # Standalone structural glue tokens (segment separator, array
        # terminator, array opener). ``"},{"`` is the speaker-turn boundary.
        for glue in ('"},{"', '"}]', '"},', "[{", '[{"', "}]"):
            try:
                ids.update(_encode(self.tokenizer, glue))
            except Exception:
                continue
        return ids

    # ------------------------------------------------------------------
    @staticmethod
    def _as_list(x: Any) -> list[int]:
        if hasattr(x, "tolist"):
            return list(x.tolist())
        return list(x)

    def __call__(self, features: list[dict]) -> dict:
        batch_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_attn: list[list[int]] = []

        for feat in features:
            ids = self._as_list(feat["input_ids"])
            labels = self._as_list(feat["labels"])
            attn = self._as_list(feat.get("attention_mask", [1] * len(ids)))
            # Left-truncate, keeping the tail: transcripts live at the end.
            if len(ids) > self.max_len:
                ids = ids[-self.max_len:]
                labels = labels[-self.max_len:]
                attn = attn[-self.max_len:]
            batch_ids.append(ids)
            batch_labels.append(labels)
            batch_attn.append(attn)

        max_len = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(features), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, (ids, lab, attn) in enumerate(zip(batch_ids, batch_labels, batch_attn)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(lab, dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(attn, dtype=torch.long)

        token_weights = build_token_weights(
            labels, self.special_token_ids, upweight=self.speaker_ts_upweight
        )

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_weights": token_weights,
        }

        if any("audio_latents" in f and f["audio_latents"] is not None for f in features):
            latents = []
            for f in features:
                al = f.get("audio_latents")
                if al is None:
                    al = torch.zeros(0, 1)
                elif not isinstance(al, torch.Tensor):
                    al = torch.as_tensor(al, dtype=torch.float32)
                latents.append(al.float())
            dim = max(al.size(-1) for al in latents if al.numel() > 0)
            t_max = max(al.size(0) for al in latents)
            padded = torch.zeros(len(latents), t_max, dim)
            mask = torch.zeros(len(latents), t_max, dtype=torch.long)
            for i, al in enumerate(latents):
                if al.numel() == 0:
                    continue
                padded[i, : al.size(0), : al.size(-1)] = al
                mask[i, : al.size(0)] = 1
            batch["audio_latents"] = padded
            batch["audio_latents_mask"] = mask

        return batch
