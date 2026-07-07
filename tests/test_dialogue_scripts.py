"""Tests for distil_vibevoice.data.dialogue_scripts (CPU-only, no network)."""

from __future__ import annotations

import json
import re

import pytest

from distil_vibevoice.data.dialogue_scripts import (
    DOMAINS,
    DialogueScript,
    Turn,
    generate_scripts,
)

_ZH_RE = re.compile(r"[一-鿿]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _all_text(script: DialogueScript) -> str:
    return " ".join(t.text for t in script.turns)


def test_deterministic_per_seed() -> None:
    a = generate_scripts(4, seed=123)
    b = generate_scripts(4, seed=123)
    assert a == b


def test_different_seeds_differ() -> None:
    a = generate_scripts(3, seed=1)
    b = generate_scripts(3, seed=2)
    assert a != b


def test_code_switching_present() -> None:
    scripts = generate_scripts(5, seed=0)
    for script in scripts:
        text = _all_text(script)
        assert _ZH_RE.search(text), "expected Chinese characters"
        assert _ASCII_WORD_RE.search(text), "expected embedded English words"


def test_structure_bounds() -> None:
    scripts = generate_scripts(10, seed=7)
    assert len(scripts) == 10
    for script in scripts:
        assert 2 <= len(script.speakers) <= 8
        assert 20 <= len(script.turns) <= 60
        assert len(set(script.speakers)) == len(script.speakers)
        assert script.domain in DOMAINS
        assert script.language == "zh-TW-en"
        for turn in script.turns:
            assert isinstance(turn, Turn)
            assert turn.speaker in script.speakers
            assert turn.text.strip()


def test_no_immediate_self_reply() -> None:
    for script in generate_scripts(3, seed=11):
        for prev, cur in zip(script.turns, script.turns[1:]):
            assert prev.speaker != cur.speaker


def test_domains_filter() -> None:
    scripts = generate_scripts(6, domains=["hiring"], seed=0)
    assert all(s.domain == "hiring" for s in scripts)
    with pytest.raises(ValueError):
        generate_scripts(1, domains=["nonexistent_domain"])


def test_llm_fn_hook_parsed() -> None:
    def fake_llm(prompt: str) -> str:
        turns = [
            {"speaker": "陳志明", "text": "我們先 review 這個 sprint 的進度。"},
            {"speaker": "Amy", "text": "好，deadline 前應該可以完成。"},
        ]
        return "Sure! Here is the meeting:\n" + json.dumps(turns, ensure_ascii=False)

    scripts = generate_scripts(2, seed=0, llm_fn=fake_llm)
    for script in scripts:
        assert [t.speaker for t in script.turns] == ["陳志明", "Amy"]
        assert script.speakers == ["陳志明", "Amy"]


def test_llm_fn_fallback_on_garbage() -> None:
    calls: list[str] = []

    def bad_llm(prompt: str) -> str:
        calls.append(prompt)
        return "I cannot help with that."

    scripts = generate_scripts(2, seed=42, llm_fn=bad_llm)
    baseline = generate_scripts(2, seed=42)
    assert len(calls) == 2
    assert scripts == baseline  # falls back to identical template output


def test_llm_fn_fallback_on_exception() -> None:
    def raising_llm(prompt: str) -> str:
        raise RuntimeError("api down")

    scripts = generate_scripts(1, seed=5, llm_fn=raising_llm)
    assert scripts == generate_scripts(1, seed=5)
