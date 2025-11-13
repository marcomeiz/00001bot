"""Simple heuristics to flag sludge/LLM-ish text."""

from __future__ import annotations

import re
from typing import Any, Dict


AI_SUSPECT_PATTERNS = [
    "in today's world",
    "in the modern world",
    "at the end of the day",
    "it's important to",
    "it's crucial to",
    "on a deeper level",
    "let that sink in",
    "game-changer",
    "unlock your potential",
    "take it to the next level",
    "crush your goals",
    "empower yourself",
    "embrace the journey",
    "ever-evolving landscape",
    "leverage",
    "harness the power of",
    "remember that",
    r"here are [0-9]+ (ways|reasons|tips)",
    r"as a (founder|leader|professional)",
]

SOFTENERS = {
    "maybe",
    "perhaps",
    "might",
    "can",
    "could",
    "often",
    "generally",
    "help you",
    "helping you",
    "ensure that",
    "strive to",
}


def ai_cop_score(text: str) -> Dict[str, Any]:
    t = text.strip()
    low = t.lower()

    hits = []
    for pattern in AI_SUSPECT_PATTERNS:
        pattern_low = pattern.lower()
        if re.search(pattern_low, low):
            hits.append(f"phrase:{pattern_low}")

    tokens = re.findall(r"\b[\w']+\b", low)
    total_tokens = len(tokens) or 1
    soft_count = 0
    for idx, tok in enumerate(tokens):
        if tok in SOFTENERS:
            soft_count += 1
            continue
        if idx < len(tokens) - 1:
            pair = f"{tok} {tokens[idx + 1]}"
            if pair in SOFTENERS:
                soft_count += 1

    soft_ratio = soft_count / total_tokens
    if soft_ratio > 0.08 and total_tokens > 20:
        hits.append(f"softeners_ratio:{soft_ratio:.3f}")

    if low.startswith(("in conclusion", "to summarize", "overall,")):
        hits.append("bloggy_closure")

    if "as a team" in low or "our mission" in low:
        hits.append("corporate_we")

    return {
        "suspect": bool(hits),
        "reasons": hits,
        "soft_ratio": soft_ratio,
    }
