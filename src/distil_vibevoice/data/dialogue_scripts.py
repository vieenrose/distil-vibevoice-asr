"""Offline generator of zh-TW/en code-switched meeting dialogue scripts.

Template-grammar based, fully deterministic per seed (no global ``random``).
Optionally accepts an ``llm_fn`` hook that is prompted for richer scripts;
its output is parsed and validated, falling back to templates on failure.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Callable

__all__ = ["Turn", "DialogueScript", "generate_scripts"]


@dataclass
class Turn:
    """A single speaker turn in a meeting dialogue."""

    speaker: str
    text: str


@dataclass
class DialogueScript:
    """A scripted multi-speaker meeting dialogue."""

    speakers: list[str]
    turns: list[Turn]
    domain: str
    language: str = "zh-TW-en"


# ---------------------------------------------------------------------------
# Grammar tables
# ---------------------------------------------------------------------------

DOMAINS: list[str] = [
    "product_standup",
    "sales_review",
    "engineering_sync",
    "budget_planning",
    "hiring",
    "marketing_campaign",
    "customer_support",
    "legal_compliance",
    "logistics_ops",
    "finance_review",
    "hr_policy",
    "partnership",
]

_ZH_SURNAMES = ["陳", "林", "王", "張", "李", "吳", "劉", "蔡", "楊", "黃",
                "許", "鄭", "謝", "郭", "洪", "曾", "邱", "廖", "賴", "周"]
_ZH_GIVEN = ["志明", "美玲", "家豪", "淑芬", "冠宇", "怡君", "承翰", "雅婷", "俊傑", "宜蓁",
             "建宏", "佩珊", "柏翰", "欣怡", "威廷", "詩涵", "彥廷", "曉婷", "宗翰", "郁婷"]
_EN_NAMES = ["Kevin", "Amy", "Jason", "Vivian", "Eric", "Peggy", "Tony", "Sandy", "Mark", "Tina",
             "Grace", "Leo", "Joyce", "Alex", "Wendy", "Sam", "Iris", "Ryan", "Nina", "Howard"]

# English words commonly embedded in zh-TW business speech.
_CS_LEXICON = [
    "deadline", "align", "timeline", "feature", "bug", "launch", "Q3", "KPI",
    "review", "merge", "schedule", "meeting", "demo", "release", "spec",
    "issue", "budget", "target", "pipeline", "offer", "headcount", "sprint",
    "roadmap", "feedback", "update", "server", "API", "app", "user", "PM",
    "OK", "case", "follow up", "sync", "onboard", "quota", "milestone",
    "campaign", "SLA", "chatbot", "NDA", "audit", "compliance", "forecast",
    "cash flow", "invoice", "PO", "SKU", "lead time", "warehouse", "vendor",
    "escalate", "stakeholder", "OKR", "one-on-one", "offsite", "workshop",
    "briefing", "proposal", "term sheet", "MOU", "royalty", "booth", "Q4",
]

_DOMAIN_TOPICS: dict[str, list[str]] = {
    "product_standup": ["新版 app 的 onboarding 流程", "會員系統改版", "推播通知功能", "首頁改版", "訂閱方案"],
    "sales_review": ["北區的業績", "新客戶的 pipeline", "續約率", "通路夥伴合作案", "年度 quota"],
    "engineering_sync": ["資料庫遷移", "API 效能問題", "CI 的 pipeline", "行動端的 crash 率", "登入服務重構"],
    "budget_planning": ["行銷預算分配", "雲端費用", "設備採購", "外包費用", "下半年的 headcount"],
    "hiring": ["後端工程師的職缺", "資深 PM 的面試", "實習生計畫", "薪資結構調整", "主管職的 offer"],
    "marketing_campaign": ["雙十一檔期的活動", "社群平台的投放策略", "KOL 合作名單", "品牌改版的 slogan", "會員回購率的 campaign"],
    "customer_support": ["客訴案件的處理流程", "退貨政策更新", "客服 chatbot 的導入", "回覆時效的 SLA", "VIP 客戶的專屬窗口"],
    "legal_compliance": ["新版個資法的因應", "合約範本更新", "資安稽核的缺失項目", "供應商的 NDA", "勞動檢查的準備"],
    "logistics_ops": ["倉儲空間的調度", "出貨延遲的改善", "第三方物流的報價", "庫存盤點的差異", "退貨入庫的 lead time"],
    "finance_review": ["月結的應收帳款", "匯率避險的部位", "股東會的財報準備", "成本分攤的原則", "資本支出的核准流程"],
    "hr_policy": ["遠端工作政策", "績效考核制度改版", "員工旅遊的規劃", "教育訓練的預算", "新人報到流程"],
    "partnership": ["日本代理商的合約", "技術授權的條件", "共同行銷的分工", "海外展會的攤位", "策略投資的意向書"],
}

_MONTHS_ZH = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"]
_WEEKDAYS_ZH = ["週一", "週二", "週三", "週四", "週五"]
_WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

_OPENINGS = [
    "大家好，我們開始今天的{domain_zh}，先請每個人快速 update 一下進度。",
    "好，人都到齊了，今天主要要 align {topic}，還有確認 {date} 的 deadline。",
    "不好意思讓大家久等，我們直接進入正題，今天的 agenda 是{topic}。",
    "各位早，這次 meeting 主要是 review {topic}，時間控制在三十分鐘內。",
]

_STATUS_UPDATES = [
    "我這邊 {topic} 的部分大概完成 {pct}%，剩下的預計 {weekday_zh} 之前可以 close。",
    "上週的 {n} 個 issue 已經處理掉 {m} 個，剩下的都跟 {topic} 有關，需要再一點時間。",
    "{topic} 目前 on track，不過 {weekday_zh} 的 demo 可能要延到 {weekday_en}。",
    "我先講一下，{topic} 的 spec 已經定稿了，這禮拜會開始進 sprint。",
    "跟大家 update 一下，{topic} 的初版昨天已經 merge 進 main 了，等 QA review。",
    "數字上來看，{topic} 這個月成長了 {pct}%，比原本的 target 高。",
]

_QUESTIONS = [
    "請問一下，{topic} 的 timeline 會不會影響到 {month_zh} 的 launch？",
    "我想確認一下，這個 feature 的 owner 是誰？deadline 是 {date} 沒錯吧？",
    "{name}，你剛剛說的 {pct}% 是包含 {topic} 嗎？",
    "所以我們的 KPI 還是維持原本的 {n} 萬嗎？還是要往上調？",
    "這個 bug 的 root cause 找到了嗎？會不會 block 到 release？",
    "預算的部分，{topic} 大概要抓多少？有 quote 了嗎？",
]

_ANSWERS = [
    "應該不會，我們有留 buffer，最壞的情況就是把 {topic} 排到下個 sprint。",
    "對，deadline 還是 {date}，不過如果 {name} 那邊的 review 來不及，可能要往後推兩天。",
    "有包含，那個數字是把 {topic} 一起算進去的。",
    "我建議先維持，等 {month_zh} 的數字出來再 review 一次。",
    "找到了，是 cache 的問題，fix 已經在跑 CI 了，不會 block release。",
    "初估大概 {n} 萬，我 {weekday_zh} 之前會把詳細的 breakdown 寄給大家。",
]

_INTERRUPTIONS = [
    "抱歉打斷一下，{topic} 這件事我覺得要先跟法務 confirm 過。",
    "等等，這個部分我補充一下，客戶那邊其實已經有 feedback 了。",
    "先暫停一下，我們時間有點不夠，這個 topic 要不要 offline 再討論？",
    "不好意思插個話，這跟上次 {name} 提的 issue 是同一個嗎？",
]

_ACTION_ITEMS = [
    "好，那 action item 就是 {name} 負責 {topic}，{date} 前給大家 update。",
    "結論是這樣：{topic} 先 hold，我們 {weekday_zh} 再開一次 sync 確認。",
    "那就這樣定案，{name} follow up {topic}，有問題隨時在群組講。",
    "OK，今天先到這邊，會議記錄我等等發出來，大家記得 review 自己的 action item。",
]

_FULL_EN_TURNS = [
    "Sorry, just to add one thing — we need to double check the numbers with finance before we commit to that timeline.",
    "Quick note: the client asked us to move the review meeting to next {weekday_en}, so please update your calendars.",
    "I'll take that one. Let me sync with the design team and get back to you by {weekday_en}.",
    "One more thing — the staging server will be down for maintenance tomorrow morning, so plan your testing around that.",
    "Can we park this for now? I don't think we have enough data to make the call today.",
]

_DOMAIN_ZH = {
    "product_standup": "產品站立會議",
    "sales_review": "業務檢討會議",
    "engineering_sync": "工程同步會議",
    "budget_planning": "預算規劃會議",
    "hiring": "招募會議",
    "marketing_campaign": "行銷企劃會議",
    "customer_support": "客服檢討會議",
    "legal_compliance": "法遵會議",
    "logistics_ops": "物流營運會議",
    "finance_review": "財務檢討會議",
    "hr_policy": "人資政策會議",
    "partnership": "合作洽談會議",
}


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _make_speakers(rng: random.Random, n: int) -> list[str]:
    """Draw ``n`` unique speaker names, mixing Chinese and English names."""
    zh = [s + g for s in _ZH_SURNAMES for g in _ZH_GIVEN]
    rng.shuffle(zh)
    en = list(_EN_NAMES)
    rng.shuffle(en)
    n_en = rng.randint(1, max(1, n // 2))
    names = en[:n_en] + zh[: n - n_en]
    rng.shuffle(names)
    return names


def _fill(template: str, rng: random.Random, domain: str, speakers: list[str]) -> str:
    """Fill a turn template with randomly drawn slot values."""
    month_i = rng.randrange(12)
    values = {
        "topic": rng.choice(_DOMAIN_TOPICS[domain]),
        "domain_zh": _DOMAIN_ZH[domain],
        "pct": rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90]),
        "n": rng.randint(2, 30),
        "m": rng.randint(1, 10),
        "date": rng.choice([f"{month_i + 1}/{rng.randint(1, 28)}", f"{_MONTHS_ZH[month_i]}{rng.randint(1, 28)}號"]),
        "month_zh": _MONTHS_ZH[month_i],
        "weekday_zh": rng.choice(_WEEKDAYS_ZH),
        "weekday_en": rng.choice(_WEEKDAYS_EN),
        "name": rng.choice(speakers),
    }
    text = template.format(**values)
    # Occasionally sprinkle an extra code-switch word as a tag phrase.
    if rng.random() < 0.25:
        text += f" 這個 {rng.choice(_CS_LEXICON)} 大家再留意一下。"
    return text


def _generate_one(rng: random.Random, domain: str) -> DialogueScript:
    """Generate a single template-based meeting script."""
    n_speakers = rng.randint(2, 8)
    speakers = _make_speakers(rng, n_speakers)
    n_turns = rng.randint(20, 60)

    turns: list[Turn] = []
    chair = speakers[0]
    turns.append(Turn(chair, _fill(rng.choice(_OPENINGS), rng, domain, speakers)))

    prev_speaker = chair
    while len(turns) < n_turns - 1:
        r = rng.random()
        if r < 0.40:
            pool = _STATUS_UPDATES
        elif r < 0.60:
            pool = _QUESTIONS
        elif r < 0.80:
            pool = _ANSWERS
        elif r < 0.90:
            pool = _INTERRUPTIONS
        else:
            pool = _FULL_EN_TURNS
        candidates = [s for s in speakers if s != prev_speaker]
        if len(turns) == n_turns - 2:
            # last loop turn: also exclude the chair, who closes the meeting next
            candidates = [s for s in candidates if s != chair] or candidates
        speaker = rng.choice(candidates)
        turns.append(Turn(speaker, _fill(rng.choice(pool), rng, domain, speakers)))
        prev_speaker = speaker

    turns.append(Turn(chair, _fill(rng.choice(_ACTION_ITEMS), rng, domain, speakers)))
    return DialogueScript(speakers=speakers, turns=turns, domain=domain)


# ---------------------------------------------------------------------------
# LLM hook
# ---------------------------------------------------------------------------

_LLM_PROMPT = (
    "You are writing a realistic Taiwanese business meeting transcript in "
    "Traditional Chinese (zh-TW) with natural English code-switching "
    "(words like deadline, align, KPI, review embedded in Chinese sentences).\n"
    "Domain: {domain}. Use {n_speakers} speakers named {speakers}.\n"
    "Write 20-60 turns. Respond ONLY with a JSON array of objects, each "
    '{{"speaker": <name>, "text": <utterance>}}. No other text.'
)


def _parse_llm_output(raw: str, domain: str) -> DialogueScript | None:
    """Parse an llm_fn response into a DialogueScript; None on failure."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list) or not data:
        return None
    turns: list[Turn] = []
    seen: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        speaker = item.get("speaker")
        text = item.get("text")
        if not isinstance(speaker, str) or not isinstance(text, str) or not text.strip():
            return None
        turns.append(Turn(speaker=speaker.strip(), text=text.strip()))
        if speaker.strip() not in seen:
            seen.append(speaker.strip())
    if len(turns) < 2 or not 2 <= len(seen) <= 12:
        return None
    return DialogueScript(speakers=seen, turns=turns, domain=domain)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_scripts(
    n: int,
    domains: list[str] | None = None,
    seed: int = 0,
    llm_fn: Callable[[str], str] | None = None,
) -> list[DialogueScript]:
    """Generate ``n`` zh-TW/en code-switched meeting dialogue scripts.

    Deterministic for a given ``(n, domains, seed)``. When ``llm_fn`` is
    given it is prompted for each script; unparseable responses fall back
    to the template grammar (which keeps the RNG stream aligned by always
    drawing the template script first).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    active_domains = list(domains) if domains else list(DOMAINS)
    for d in active_domains:
        if d not in _DOMAIN_TOPICS:
            raise ValueError(f"unknown domain: {d!r} (known: {sorted(_DOMAIN_TOPICS)})")

    rng = random.Random(seed)
    scripts: list[DialogueScript] = []
    for i in range(n):
        domain = active_domains[i % len(active_domains)]
        template_script = _generate_one(rng, domain)
        if llm_fn is not None:
            prompt = _LLM_PROMPT.format(
                domain=domain,
                n_speakers=len(template_script.speakers),
                speakers=", ".join(template_script.speakers),
            )
            try:
                raw = llm_fn(prompt)
            except Exception:
                raw = ""
            parsed = _parse_llm_output(raw, domain) if raw else None
            scripts.append(parsed if parsed is not None else template_script)
        else:
            scripts.append(template_script)
    return scripts
