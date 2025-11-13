"""Quick A/B calibration runner for VoiceEngine."""

from __future__ import annotations

from typing import List, Dict

from marco_voice_engine.pipeline import VoiceEngine

IDEAS: List[str] = [
    "You worked 60 hours last week and nothing moved.",
    "If you don't log decisions, you'll re-argue the same shit every quarter.",
    "Stop doing free audits for people who never had budget.",
    "You don't need more leads, you need to stop losing the ones you already paid for.",
    "Protect 2 days a week with zero calls or you will never build anything real.",
    "If your ops live in your head, you're the bottleneck, not the hero.",
    "Client pays late? Your process is vague, not their ethics.",
    "If every week is 'exceptional', your system is shit, not your calendar.",
    "You confuse motion with progress because chaos makes you feel important.",
    "A solo founder that never says no is just an underpaid employee.",
    "Kill one channel, one offer, one ICP for 90 days and see what happens.",
    "Your team is not slow, your briefs are.",
    "If you can't explain how cash moves in 5 lines, you don't run a business.",
    "Every new tool you add without deleting one roba foco.",
    "If you need a crisis to move, you'll manufacture drama without noticing.",
    "Hablas de libertad pero vives esclavo de Slack.",
    "Tu agenda llena es tu dopamina, no tu negocio.",
    "Si todo es urgente, nadie sabe qué es importante.",
    "Delega entregables, no responsabilidad.",
    "El funnel roto casi nunca es tráfico, es onboarding.",
]


def main() -> None:
    engine = VoiceEngine()
    totals: Dict[str, Dict[str, int]] = {
        "A": {"gen": 0, "ok": 0},
        "B": {"gen": 0, "ok": 0},
        "A_retry": {"gen": 0, "ok": 0},
        "B_retry": {"gen": 0, "ok": 0},
    }

    for idea in IDEAS:
        result = engine.propose(idea, retry_on_empty=True)

        print("\n=== IDEA ===")
        print(idea)

        print("\nACCEPTED:")
        if not result["accepted"]:
            print("- None")
        for item in result["accepted"]:
            tag = item.get("model", "?")
            sim = item["scores"].get("similarity", 0.0)
            print(f"[{tag}] {sim:.3f} | {item['text']}")
            if tag in totals:
                totals[tag]["ok"] += 1
                totals[tag]["gen"] += 1

        print("\nREJECTED:")
        for item in result["rejected"]:
            tag = item.get("model", "?")
            reasons = ",".join(item["reasons"]) or "-"
            sim = item["scores"].get("similarity", 0.0)
            print(f"[{tag}] {sim:.3f} | {item['text']} || {reasons}")
            if tag in totals:
                totals[tag]["gen"] += 1

    print("\n=== SUMMARY ===")
    for tag, stats in totals.items():
        if stats["gen"] == 0:
            continue
        ratio = stats["ok"] / stats["gen"] if stats["gen"] else 0.0
        print(f"{tag}: {stats['ok']} / {stats['gen']} accepted ({ratio:.2%})")


if __name__ == "__main__":
    main()
