"""CPU-only tests for DistillCollator (no transformers, no network).

A tiny deterministic character-level tokenizer stands in for the real
Qwen2.5 tokenizer; it exposes the subset of the HF tokenizer API that the
collator relies on.
"""

from __future__ import annotations

import torch

from distil_vibevoice.distill.collator import DistillCollator


class CharTokenizer:
    """Deterministic char-level tokenizer with a stable vocabulary."""

    pad_token_id = 0
    eos_token_id = 1

    _OFFSET = 2  # ids 0/1 reserved for pad/eos

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text)}

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 4000 + self._OFFSET for c in text]


def _id(char: str) -> int:
    return ord(char) % 4000 + CharTokenizer._OFFSET


def test_special_ids_derived_from_markup() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64)
    # The markup delimiters ("Start":, "End":, "Speaker":, "Content":) must
    # yield a non-empty id set that includes the quote/colon and key letters.
    assert coll.special_token_ids
    assert _id('"') in coll.special_token_ids
    assert _id(":") in coll.special_token_ids
    assert _id("S") in coll.special_token_ids  # from "Start"/"Speaker"


def test_padding_shapes_and_values() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64)
    f1 = {"input_ids": [10, 11, 12], "labels": [10, 11, 12]}
    f2 = {"input_ids": [20, 21, 22, 23, 24], "labels": [20, 21, 22, 23, 24]}
    batch = coll([f1, f2])

    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["token_weights"].shape == (2, 5)

    # Right padding with pad id / -100 / 0.
    assert batch["input_ids"][0].tolist() == [10, 11, 12, 0, 0]
    assert batch["labels"][0].tolist() == [10, 11, 12, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    # Weights are zero on padding (label -100).
    assert batch["token_weights"][0].tolist()[3:] == [0.0, 0.0]


def test_left_truncation_keeps_tail() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=4)
    ids = list(range(100, 110))  # 10 tokens, cap is 4
    batch = coll([{"input_ids": ids, "labels": ids}])
    assert batch["input_ids"].shape == (1, 4)
    assert batch["input_ids"][0].tolist() == [106, 107, 108, 109]
    assert batch["labels"][0].tolist() == [106, 107, 108, 109]


def test_token_weights_align_with_labels() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64, speaker_ts_upweight=4.0)
    special = next(iter(coll.special_token_ids))
    plain = max(coll.special_token_ids) + 1000  # guaranteed non-special
    plain2 = plain + 1
    assert plain not in coll.special_token_ids and plain2 not in coll.special_token_ids

    feats = [{"input_ids": [plain, special, plain, plain2], "labels": [plain, special, plain, plain2]}]
    weights = coll(feats)["token_weights"][0]
    # special position and the one after it get the upweight, rest are 1.
    assert weights.tolist() == [1.0, 4.0, 4.0, 1.0]


def test_labels_minus_100_positions_get_zero_weight() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64)
    feats = [{"input_ids": [5, 6, 7], "labels": [-100, -100, 7]}]
    weights = coll(feats)["token_weights"][0]
    assert weights[0] == 0.0 and weights[1] == 0.0 and weights[2] > 0.0


class MergeTokenizer:
    """Greedy longest-match tokenizer mimicking the real Qwen2.5 BPE merges of
    the structured target's punctuation.

    Like the real 152k-vocab tokenizer, the JSON glue merges into multi-char
    tokens whose id depends on surrounding context — in particular the
    inter-segment / speaker-turn boundary ``"},{"`` is a single id that the
    open-only context prefixes never produce. Ids are arbitrary but stable.
    """

    pad_token_id = 0
    eos_token_id = 1

    # Longest-first so greedy matching prefers merged structural tokens.
    _VOCAB = {
        '"},{"': 900,  # object close + separator + next open (the boundary)
        '"}],{': 915,
        '"}]': 916,
        '[{"': 917,
        '"},': 918,
        '{"': 901,
        ',{"': 902,
        ',"': 903,
        '":': 904,
        '"}': 905,
        "[{": 906,
        "],": 919,
        "]": 907,
        "[": 908,
        "{": 909,
        "}": 910,
        ",": 911,
        '"': 912,
        "Start": 921,
        "End": 922,
        "Speaker": 923,
        "Content": 924,
    }

    def __init__(self) -> None:
        self._keys = sorted(self._VOCAB, key=len, reverse=True)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text)}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            for k in self._keys:
                if text.startswith(k, i):
                    ids.append(self._VOCAB[k])
                    i += len(k)
                    break
            else:
                ids.append(ord(text[i]) % 800 + 2000)  # non-colliding fallback
                i += 1
        return ids


def test_separator_token_upweighted_with_bpe_merges() -> None:
    """The fix: markup detection must land on the ``"},{"`` boundary token.

    With BPE-style merges the speaker-turn boundary is a single id that the
    old open-context-only variants (``{`` / ``,`` / ``,{`` prefixes) never
    produced. It must now be in the derived special set and get upweighted.
    """
    coll = DistillCollator(MergeTokenizer(), max_len=64)
    sep_id = MergeTokenizer._VOCAB['"},{"']
    assert sep_id in coll.special_token_ids, "segment-separator token not detected"
    # Keys and open-context glue are still covered.
    for tok_str in ("Start", "End", "Speaker", "Content", '{"', ',"', '":'):
        assert MergeTokenizer._VOCAB[tok_str] in coll.special_token_ids

    # A two-segment stream: the boundary label must receive the 4x weight.
    ids = MergeTokenizer().encode(
        '[{"Start":0,"End":1,"Speaker":0,"Content":"a"},{"Start":1,"End":2,"Speaker":1,"Content":"b"}]'
    )
    w = coll([{"input_ids": ids, "labels": ids}])["token_weights"][0].tolist()
    sep_pos = ids.index(sep_id)
    assert w[sep_pos] == 4.0


def test_audio_latents_padded_and_masked() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64)
    f1 = {
        "input_ids": [3, 4],
        "labels": [3, 4],
        "audio_latents": torch.randn(3, 4),
    }
    f2 = {
        "input_ids": [5, 6, 7],
        "labels": [5, 6, 7],
        "audio_latents": torch.randn(5, 4),
    }
    batch = coll([f1, f2])
    assert batch["audio_latents"].shape == (2, 5, 4)
    assert batch["audio_latents_mask"].shape == (2, 5)
    assert batch["audio_latents_mask"].sum(dim=1).tolist() == [3, 5]
    # Padded region is zeros.
    assert torch.all(batch["audio_latents"][0, 3:] == 0)


def test_no_audio_latents_key_when_absent() -> None:
    coll = DistillCollator(CharTokenizer(), max_len=64)
    batch = coll([{"input_ids": [3, 4], "labels": [3, 4]}])
    assert "audio_latents" not in batch
    assert "audio_latents_mask" not in batch
