"""CLI para ejecutar el pipeline rápidamente."""

from __future__ import annotations

import sys

from .pipeline import VoiceEngine


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python -m marco_voice_engine "your idea"')
        print('  python -m marco_voice_engine chaos "your idea"')
        raise SystemExit(1)

    first = sys.argv[1].lower()
    if first == "chaos":
        mode = "chaos"
        if len(sys.argv) < 3:
            print("Missing idea for chaos mode.")
            raise SystemExit(1)
        idea = sys.argv[2]
    else:
        mode = "ops"
        idea = sys.argv[1]

    engine = VoiceEngine(mode=mode)
    result = engine.propose(idea)

    print(f"\nMODE: {mode}")
    print("IDEA:", result["idea"])

    print("\nACCEPTED:")
    accepted = result.get("accepted", [])
    if not accepted:
        print("- None accepted")
    else:
        for idx, item in enumerate(accepted, start=1):
            sim = item.get("scores", {}).get("similarity", 0.0)
            print(f"\n#{idx} (sim={sim:.3f}):")
            print("```")
            print(item["text"].strip())
            print("```")

    print("\nREJECTED (compact):")
    rejected = result.get("rejected", [])
    if not rejected:
        print("- None")
    else:
        for item in rejected[:5]:
            reasons = ",".join(item.get("reasons", [])) or "-"
            sim = item.get("scores", {}).get("similarity", 0.0)
            print(f"- sim={sim:.3f} | {item['text']} || {reasons}")


if __name__ == "__main__":
    main()
