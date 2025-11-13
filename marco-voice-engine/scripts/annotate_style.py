"""Annotate dataset.json with heuristic style labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

OPS_KEYWORDS = {
    "founder",
    "founders",
    "startup",
    "startups",
    "mrr",
    "sass",
    "saas",
    "agent",
    "agents",
    "infra",
    "product",
    "ship",
    "shipping",
    "build",
    "builder",
    "builders",
    "company",
    "companies",
    "sales",
    "deal",
    "deals",
    "ops",
    "operating",
    "ai",
    "llm",
    "tooling",
    "system",
    "systems",
}

CHAOS_KEYWORDS = {
    "waifu",
    "waifus",
    "horny",
    "brainrot",
    "grok",
    "anime",
    "shitpost",
    "shitposting",
    "touch grass",
    "ketamine",
    "doomscroll",
    "doomscrolling",
    "villain",
    "nihilism",
    "unhinged",
}

ALLOWED_STYLES = {"ops", "chaos", "both", "neutral"}


def detect_style(text: str) -> str:
    """Classify text into one of the style buckets."""

    lower = text.lower()
    ops_hits = any(word in lower for word in OPS_KEYWORDS)
    chaos_hits = any(word in lower for word in CHAOS_KEYWORDS)

    if ops_hits and chaos_hits:
        return "both"
    if ops_hits:
        return "ops"
    if chaos_hits:
        return "chaos"
    return "neutral"


def annotate_dataset(dataset_path: Path) -> None:
    """Load dataset.json, add style per entry, and overwrite."""

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    tweets: List[Dict] = data.get("tweets", [])
    updated = []

    for entry in tweets:
        style = entry.get("style")
        if style not in ALLOWED_STYLES:
            style = detect_style(entry.get("text", ""))
        entry["style"] = style
        updated.append(entry)

    data["tweets"] = updated
    dataset_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root.parent / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.json not found at: {dataset_path}")
    annotate_dataset(dataset_path)
    print(f"Annotated styles written back to {dataset_path}")


if __name__ == "__main__":
    main()
