"""Tests for mixed error rate (MER) and mixed zh/en tokenization."""

from __future__ import annotations

from distil_vibevoice.eval.mer import levenshtein, mer, tokenize_mixed


def test_tokenize_cjk_chars_as_single_tokens() -> None:
    assert tokenize_mixed("今天天氣很好") == ["今", "天", "天", "氣", "很", "好"]


def test_tokenize_mixed_zh_en_digits() -> None:
    assert tokenize_mixed("今天開 sprint 23 的 review") == [
        "今",
        "天",
        "開",
        "sprint",
        "23",
        "的",
        "review",
    ]


def test_tokenize_lowercases_and_strips_punct() -> None:
    assert tokenize_mixed("Hello, World! 你好。") == ["hello", "world", "你", "好"]


def test_tokenize_fullwidth_folded() -> None:
    # NFKC folds fullwidth latin/digits to ASCII before tokenizing.
    assert tokenize_mixed("ＡＢＣ　１２３") == ["abc", "123"]


def test_tokenize_empty() -> None:
    assert tokenize_mixed("") == []
    assert tokenize_mixed("，。！？ ...") == []


def test_tokenize_zero_ideograph_and_ext_b() -> None:
    # 〇 (U+3007, common in zh-TW dates) is one CJK token.
    assert tokenize_mixed("二〇二五年") == ["二", "〇", "二", "五", "年"]
    # CJK Extension B+ ideographs (e.g. 𠮟 U+20B9F in rare-character names).
    assert tokenize_mixed("𠮟責他") == ["𠮟", "責", "他"]


def test_tokenize_compatibility_ideographs() -> None:
    # U+F900 NFKC-folds to the URO form U+8C48 and must still tokenize.
    assert tokenize_mixed("\uf900") == ["豈"]  # input is the U+F900 compat char


def test_multiplication_division_signs_are_separators() -> None:
    # × (U+00D7) and ÷ (U+00F7) are punctuation, not word tokens.
    assert tokenize_mixed("3×4") == ["3", "4"]
    assert tokenize_mixed("10÷2") == ["10", "2"]
    assert mer("3×4", "3 4") == 0.0


def test_mer_counts_zero_ideograph_substitution() -> None:
    # 〇 -> 零 is exactly one substitution over five reference tokens.
    assert mer("二〇二五年", "二零二五年") == 0.2


def test_mer_spacing_insensitive_for_zh() -> None:
    assert mer("今天 天氣 很好", "今天天氣很好") == 0.0


def test_mer_perfect_is_zero() -> None:
    assert mer("大家好 we start now", "大家好 we start now") == 0.0


def test_mer_all_wrong_is_one() -> None:
    assert mer("今天", "明日") == 1.0


def test_mer_en_counted_per_word_not_char() -> None:
    # One wrong word out of two -> 0.5 (char-level would be far smaller).
    assert mer("hello world", "hello word") == 0.5


def test_mer_zh_counted_per_char() -> None:
    # One wrong char out of four.
    assert mer("今天天氣", "今天天期") == 0.25


def test_mer_punctuation_ignored() -> None:
    assert mer("你好。", "你好") == 0.0
    assert mer("OK, let's go!", "ok let's go") == 0.0


def test_mer_insertion_and_deletion() -> None:
    # Insertion: 1 extra token over 2 ref tokens.
    assert mer("你好", "你好嗎") == 0.5
    # Deletion: 1 missing token over 3 ref tokens.
    assert abs(mer("你好嗎", "你好") - 1.0 / 3.0) < 1e-12


def test_mer_empty_edge_cases() -> None:
    assert mer("", "") == 0.0
    assert mer("", "hello") == 1.0
    assert mer("你好", "") == 1.0


def test_mer_can_exceed_one() -> None:
    assert mer("好", "this is a long insertion") > 1.0


def test_levenshtein_matches_reference_dp() -> None:
    def slow(a: list[str], b: list[str]) -> int:
        d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            d[i][0] = i
        for j in range(len(b) + 1):
            d[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                )
        return d[len(a)][len(b)]

    import random

    rng = random.Random(0)
    alphabet = ["今", "天", "氣", "好", "the", "cat", "42"]
    for _ in range(50):
        a = [rng.choice(alphabet) for _ in range(rng.randrange(0, 12))]
        b = [rng.choice(alphabet) for _ in range(rng.randrange(0, 12))]
        assert levenshtein(a, b) == slow(a, b), (a, b)


def test_levenshtein_empty() -> None:
    assert levenshtein([], []) == 0
    assert levenshtein(["a"], []) == 1
    assert levenshtein([], ["a", "b"]) == 2
