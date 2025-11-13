"""Quick probe runner that samples topics and inspects outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path

from marco_voice_engine.pipeline import VoiceEngine
from marco_voice_engine.config import get_voice_data_dir, get_topics_filename


def load_topics() -> list[str]:
    """Carga topics.json como lista de strings."""

    base_dir = get_voice_data_dir()
    topics_path = base_dir / get_topics_filename()

    if not topics_path.exists():
        raise FileNotFoundError(f"topics.json not found at: {topics_path}")

    with topics_path.open("r", encoding="utf-8") as f_handle:
        data = json.load(f_handle)

    topics: list[str] = []

    def _extract_from_list(items: list[object]) -> None:
        for item in items:
            if isinstance(item, str):
                val = item.strip()
                if val:
                    topics.append(val)
            elif isinstance(item, dict):
                val = (
                    item.get("idea")
                    or item.get("topic")
                    or item.get("text")
                    or item.get("theme")
                )
                if isinstance(val, str):
                    val = val.strip()
                    if val:
                        topics.append(val)

    if isinstance(data, list):
        _extract_from_list(data)
    elif isinstance(data, dict):
        for key in ("topics", "themes"):
            maybe = data.get(key)
            if isinstance(maybe, list):
                _extract_from_list(maybe)

    topics = [t for t in topics if t]
    if not topics:
        raise ValueError("No usable topics found in topics.json")

    return topics


def main() -> None:
    topics = load_topics()
    ve = VoiceEngine()

    sample = random.sample(topics, k=min(3, len(topics)))

    for idea in sample:
        print("\n" + "=" * 80)
        print("TOPIC IDEA:", idea)
        print("=" * 80)

        result = ve.propose(idea, retry_on_empty=True)

        print("\nACCEPTED:")
        if not result["accepted"]:
            print("- None accepted")
        else:
            for idx, item in enumerate(result["accepted"], start=1):
                txt = item["text"].replace("\n", " ")
                sim = item["scores"].get("similarity", 0.0)
                print(f"[{idx}] sim={sim:.3f} | {txt}")

        print("\nREJECTED (compact view):")
        for item in result["rejected"][:3]:
            txt = item["text"].replace("\n", " ")
            reasons = ",".join(item["reasons"]) or "-"
            sim = item["scores"].get("similarity", 0.0)
            print(f"sim={sim:.3f} | {txt} || {reasons}")


if __name__ == "__main__":
    main()
