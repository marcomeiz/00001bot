"""Debug script to inspect VoiceEngine outputs + judge decisions."""

from __future__ import annotations

from marco_voice_engine.pipeline import VoiceEngine

IDEAS = [
    "You worked 60 hours last week and nothing moved.",
    "If you don't log decisions, you'll re-argue the same shit every quarter.",
    "Stop doing free audits for people who never had budget.",
]


def main() -> None:
    ve = VoiceEngine()

    for idea in IDEAS:
        print("\n" + "=" * 80)
        print("IDEA:", idea)
        print("=" * 80)

        result = ve.propose(idea, retry_on_empty=False)
        all_evals = result["accepted"] + result["rejected"]

        print("\nRAW VARIANTS + JUDGE")
        if not all_evals:
            print("No variants generated at all.")
            continue

        for idx, evaluation in enumerate(all_evals, start=1):
            text = evaluation["text"].replace("\n", " ")
            scores = evaluation.get("scores", {})
            reasons = evaluation.get("reasons", [])
            print(f"\n[{idx}]")
            print(f"TEXT: {text}")
            print(
                "len={length}, sim={sim:.3f}, length_ok={length_ok}, "
                "style_basic_ok={style_basic_ok}, too_generic={too_generic}, "
                "too_close={too_close}".format(
                    length=len(text),
                    sim=scores.get("similarity", 0.0),
                    length_ok=scores.get("length_ok"),
                    style_basic_ok=scores.get("style_basic_ok"),
                    too_generic=scores.get("too_generic"),
                    too_close=scores.get("too_close"),
                )
            )
            print(f"reasons={reasons}")
            print(f"accepted={evaluation['accepted']}")


if __name__ == "__main__":
    main()
