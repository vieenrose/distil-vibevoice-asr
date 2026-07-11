"""Conservative inverse text normalization (ITN) for zh-TW meeting transcripts.

Deterministic post-processing: spoken numbers -> digits, percentages, common
date/time forms — WITHOUT touching multi-character idioms or names where the
digits would be wrong. An acoustic model can't reliably learn Chinese place
value, so this is a separate rule module (cf. Luigi/x-asr-zh-tw-native-itn's
conservative_itn). Recognition is unchanged; only the written form improves.

itn(text) -> text with numbers normalized.
"""
from __future__ import annotations

import re

import cn2an

# Idioms / fixed expressions containing number chars whose digits would be
# wrong or jarring — left untouched. Extend as needed.
_IDIOM_GUARD = (
    "千萬", "萬一", "萬分", "萬能", "萬歲", "萬全", "萬事", "萬象", "一二",
    "三三兩兩", "五花八門", "七上八下", "九牛一毛", "五湖四海", "十全十美",
    "亂七八糟", "十之八九", "一心一意", "三心二意", "四面八方", "百分之百",
    "一五一十", "三言兩語", "百年", "千年", "萬年", "一一", "十字", "百般",
    "千方百計", "成千上萬", "億萬",
)

# Percentages: 百分之五十 -> 50%, 百分之二點五 -> 2.5%
_PCT_RE = re.compile(r"百分之([零一二三四五六七八九十百千點兩0-9]+)")

# A run of Chinese number characters (candidate for digit conversion). We keep
# the run bounded and require it to be a plausible standalone quantity.
_NUM_RE = re.compile(r"[零一二三四五六七八九十百千萬億兩]+")

# Runs to skip converting: pure "十/百/千/萬/億/兩" with no unit digits read
# more naturally as words in isolation is NOT our concern here; we DO convert
# quantities but guard idioms and ordinal-ish single chars.
_SKIP_SINGLE = set("零十百千萬億兩一二三四五六七八九")


def _to_digits(cn: str) -> str | None:
    try:
        val = cn2an.cn2an(cn, mode="smart")
    except (ValueError, KeyError):
        return None
    # cn2an returns int/float/Decimal; render cleanly
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    s = str(val)
    return s


def _guarded(text: str, start: int, end: int) -> bool:
    """True if the [start,end) span overlaps a guarded idiom."""
    lo = max(0, start - 3)
    window = text[lo:end + 3]
    frag = text[start:end]
    for idi in _IDIOM_GUARD:
        if idi in window and frag in idi:
            return True
    return False


def itn(text: str) -> str:
    if not text:
        return text

    # 1) percentages first (consume the 百分之 prefix)
    def _pct(m: re.Match) -> str:
        d = _to_digits(m.group(1))
        return f"{d}%" if d is not None else m.group(0)

    text = _PCT_RE.sub(_pct, text)

    # 2) standalone number runs -> digits, guarding idioms and lone chars
    out = []
    last = 0
    for m in _NUM_RE.finditer(text):
        s, e = m.span()
        run = m.group(0)
        out.append(text[last:s])
        last = e
        if len(run) == 1 and run in _SKIP_SINGLE:
            out.append(run)          # lone 十/一/兩… keep as-is
            continue
        if _guarded(text, s, e):
            out.append(run)
            continue
        d = _to_digits(run)
        # only substitute when the result is genuinely numeric and not absurd
        if d is not None and re.fullmatch(r"-?\d+(\.\d+)?", d):
            out.append(d)
        else:
            out.append(run)
    out.append(text[last:])
    return "".join(out)
