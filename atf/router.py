"""Adaptive inference LOD: prompt difficulty scoring + tier routing.

score_prompt(text) -> (difficulty in [0,1], features dict)
tier_for(difficulty) -> Tier dataclass with generation overrides.

Pure heuristics, no model calls -- microseconds per request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BASE = 0.35

GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|sup|good\s(morning|afternoon|evening)|"
    r"thanks?( you)?|thank you|bye+|goodbye|ok(ay)?|cool|nice)\b[\s!.?]*$",
    re.IGNORECASE)
FACT_RE = re.compile(
    r"\b(who|what|when|where)('s| is| was| are| were)\b|\bcapital of\b",
    re.IGNORECASE)
FACT_Q_RE = re.compile(
    r"^(who|what|when|where|which|how many|how much)\b.*\?", re.IGNORECASE)
MATH_SYM_RE = re.compile(r"[=+\-*/^%]|\d\s*[\^%]")
NUMBERS_RE = re.compile(r"\d")
MULTISTEP_RE = re.compile(
    r"\b(step by step|step-by-step|walk me through|solve|show that|"
    r"explain step by step)\b", re.IGNORECASE)
PROOF_RE = re.compile(
    r"\b(prove|derive|proof|induction|axiom|derivation|theorem)\b",
    re.IGNORECASE)
CODE_RE = re.compile(
    r"```|\b(function|def |class |import |const |var |let |return|"
    r"for\s*\(|while\s*\(|npm |pip |git |bash|shell|script|one-liner|"
    r"regex|python|javascript|rust|sql)\b",
    re.IGNORECASE)
CODE_COMPLEX_RE = re.compile(
    r"\b(architecture|design pattern|concurrency|concurren|thread|async|"
    r"parallel|optimize|optimiz|scalab|distributed|compiler|parser|"
    r"interpreter|refactor|memory leak|race condition|deadlock|"
    r"big o|time complexity|space complexity|data structure|"
    r"binary search tree|dynamic programming|graph algorithm|"
    r"microservice|kubernetes|database schema|migrat)\b", re.IGNORECASE)
MATH_LEXICON_RE = re.compile(
    r"\b(integral|derivative|matrix|vector|probability|algorithm|"
r"equation|theorem|calculus|algebra|geometry|statistics|"
    r"complexity|recursion|logarithm|fraction|irrational|rational|"
    r"derivation|inequality|integral)\b", re.IGNORECASE)


@dataclass
class Tier:
    level: int
    name: str
    thinking: str      # off | low | medium | high
    temperature: float
    max_tokens: int
    exact_ffn: bool


TIERS = [
    Tier(0, "instant",  "off",    0.0, 128,  True),
    Tier(1, "chat",     "off",    0.5, 1024, True),
    Tier(2, "reasoning","low",    0.6, 2048, True),
    Tier(3, "deep",     "medium", 0.7, 4096, True),
]

BADGES = {0: "⚡ instant", 1: "💬 chat", 2: "🧠 reasoning", 3: "🔬 deep"}


def score_prompt(text: str) -> tuple[float, dict]:
    """Return (difficulty in [0,1], feature breakdown)."""
    f: dict = {}
    s = BASE

    words = text.split()
    n_words = len(words)
    f["words"] = n_words

    # --- reflex signals -------------------------------------------------
    if GREETING_RE.search(text.strip()):
        s -= 0.30
        f["greeting"] = True
    if FACT_RE.search(text):
        s -= 0.15
        f["fact_pattern"] = True
    is_question = bool(FACT_RE.search(text) or text.strip().endswith("?") or re.match(
        r"^(how|why|what|where|when|who|explain|tell|describe|compare|list)",
        text.strip(), re.IGNORECASE))
    if n_words <= 6 and not NUMBERS_RE.search(text) and not is_question:
        s -= 0.20
        f["very_short"] = True
    elif FACT_RE.search(text):
        s -= 0.25
        f["short_fact"] = True

    # --- reasoning signals (diminishing returns, capped) -----------------
    hits = []
    bonus = 0.0
    def add(w, label):
        nonlocal bonus
        bonus += w
        hits.append(label)
    if PROOF_RE.search(text):
        add(0.30, "proof")
    if MATH_SYM_RE.search(text) and NUMBERS_RE.search(text):
        add(0.20, "arithmetic")
    elif MATH_SYM_RE.search(text):
        add(0.15, "math_symbols")
    if MULTISTEP_RE.search(text):
        add(0.15, "multistep")
    if CODE_COMPLEX_RE.search(text):
        add(0.40, "code_complex")
    elif CODE_RE.search(text):
        add(0.20, "code")
    if MATH_LEXICON_RE.search(text):
        add(0.15, "math_lexicon")
    if n_words > 150:
        add(0.10, "long")
    if text.count("?") >= 2 or len([l for l in text.splitlines() if l.strip()]) >= 3:
        add(0.10, "multi_part")
    s += min(bonus, 0.55)
    f["signals"] = hits
    f["bonus_capped"] = bonus > 0.55

    return max(0.0, min(1.0, s)), f


def tier_for(difficulty: float) -> Tier:
    if difficulty < 0.20:
        return TIERS[0]
    if difficulty < 0.45:
        return TIERS[1]
    if difficulty < 0.75:
        return TIERS[2]
    return TIERS[3]


def route(message: str, user_override: str | None = None) -> tuple[Tier, float]:
    """Route a message to a tier.

    user_override: None (auto) or a thinking level ("off"/"low"/...).
    Returns (tier, difficulty). Overrides keep the scored tier's compute
    profile but force the thinking level.
    """
    d, _ = score_prompt(message)
    tier = tier_for(d)
    if user_override:
        forced = user_override.strip().lower()
        if forced != "auto":
            temp_by_level = {"off": 0.0, "low": 0.6, "medium": 0.7, "high": 0.75}
            maxtok_by_level = {"off": tier.max_tokens, "low": 1024,
                               "medium": 1536, "high": 2048}
            tier = Tier(tier.level, tier.name, forced,
                        temp_by_level.get(forced, 0.7),
                        maxtok_by_level.get(forced, 1024), tier.exact_ffn)
    return tier, d


def looks_degenerate(text: str) -> bool:
    """Post-hoc check on a completed reply: empty, cut mid-sentence, or
    immediate single-token repetition."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    tail = stripped[-24:]
    if len(set(tail)) <= 2 and len(tail) >= 12:   # e.g. "55555555555..."
        return True
    if len(stripped) < 8:
        return True
    return False
