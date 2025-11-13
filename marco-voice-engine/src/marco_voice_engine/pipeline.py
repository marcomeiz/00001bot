"""Orquesta generación y evaluación según el modo (ops/chaos)."""

from __future__ import annotations

from typing import Any, Dict, List

from .generator import Generator
from .judge import Judge
from .config import get_generation_model_primary, get_generation_model_secondary
from .embeddings import get_embedding, EmbeddingError
from .reply_filter import ReplyFilter


class VoiceEngine:
    """
    Ejecuta generación y evalúa con el juez para un modo específico.
    """

    def __init__(self, mode: str = "ops") -> None:
        self._mode = mode
        primary = get_generation_model_primary()
        if not primary:
            raise RuntimeError("GENERATION_MODEL_PRIMARY not set.")

        secondary = get_generation_model_secondary()

        self._generators: List[tuple[str, Generator]] = [
            ("primary", Generator(model_name=primary, mode=self._mode))
        ]
        if secondary:
            self._generators.append(
                ("secondary", Generator(model_name=secondary, mode=self._mode))
            )

        self._judge = Judge()

    def propose(self, idea: str, retry_on_empty: bool = True) -> Dict[str, Any]:
        if not idea or not idea.strip():
            raise ValueError("Idea cannot be empty.")

        all_accepted: List[Dict[str, Any]] = []
        all_rejected: List[Dict[str, Any]] = []

        def _run_once(slot: str, generator: Generator, retry: bool = False) -> None:
            raw_variants = generator.generate_variants(idea)
            evaluated = [
                self._judge.evaluate_variant(v, mode=self._mode) for v in raw_variants
            ]
            if slot == "primary":
                tag_base = self._mode
            else:
                tag_base = f"{self._mode}_{slot}"
            tag = f"{tag_base}_retry" if retry else tag_base
            for record in evaluated:
                record["model"] = tag
                if record["accepted"]:
                    all_accepted.append(record)
                else:
                    all_rejected.append(record)

        for slot, generator in self._generators:
            _run_once(slot, generator)

        if not all_accepted and retry_on_empty:
            primary_generator = self._generators[0][1]
            _run_once("primary", primary_generator, retry=True)

        return {
            "idea": idea,
            "accepted": all_accepted,
            "rejected": all_rejected,
        }

    def reply(self, original_tweet: str) -> Dict[str, Any]:
        if not original_tweet or not original_tweet.strip():
            raise ValueError("Empty original tweet for reply().")

        reply_filter = ReplyFilter()
        filter_result = reply_filter.evaluate(original_tweet)
        result_payload: Dict[str, Any] = {
            "original": original_tweet.strip(),
            "filter": filter_result.to_dict(),
        }
        if not filter_result.accepted:
            result_payload["accepted"] = []
            result_payload["rejected"] = []
            return result_payload

        context_embedding = None
        stripped_original = original_tweet.strip()
        try:
            context_embedding = get_embedding(stripped_original)
        except EmbeddingError:
            context_embedding = None

        generator = Generator(
            model_name=get_generation_model_primary(),
            mode="reply",
        )
        judge = Judge()

        raw_variants = generator.generate_variants(original_tweet)
        evaluated = [
            judge.evaluate_variant(
                v,
                mode="reply",
                context=stripped_original,
                context_embedding=context_embedding,
            )
            for v in raw_variants
        ]

        accepted = [record for record in evaluated if record["accepted"]]
        rejected = [record for record in evaluated if not record["accepted"]]

        result_payload["accepted"] = accepted[:3]
        result_payload["rejected"] = rejected
        return result_payload
