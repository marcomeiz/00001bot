"""Filtro de estilo/voz que evalúa variantes generadas."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

import numpy as np

from .vector_store import VectorStore
from .embeddings import get_embedding, EmbeddingError
from .ai_cop import ai_cop_score

MIN_LEN = 40
MAX_LEN = 280
MODE_LENGTH_LIMITS = {
    "reply": (20, 220),
}

MIN_SIM = 0.35
MAX_SIM = 0.97

REPLY_MIN_CONTEXT_SIM = 0.32
REPLY_MAX_CONTEXT_SIM = 0.92

REPLY_BANNED_HOOKS = {
    "this hits",
    "yep",
    "oof",
    "love this",
}

BANNED_SUBSTRINGS = [
    "as an ai",
    "as a language model",
    "chatgpt",
    "hustle",
    "crush your goals",
    "mindset de éxito",
    "emprende o muere",
    "querido fundador",
    "inspirational quote",
    "build something real",
    "that's when the real building happens",
    "guard this time like it's gold",
    "focus on real results",
    "if you want to build something real",
    "you feel important, but you're not getting anywhere",
]

BANNED_PREFIXES = [
    "here are",
    "in this thread",
    "as a founder you must",
    "debes entender que",
]

CHECKLIST_PATTERN = re.compile(r"^(?:\d+\.|\- )", re.IGNORECASE)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    va = np.array(vec_a, dtype="float32")
    vb = np.array(vec_b, dtype="float32")
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class Judge:
    """
    Evalúa variantes generadas y decide si pasan el filtro de voz/estilo.
    No genera. No publica.
    """

    def __init__(self) -> None:
        self._vs = VectorStore.shared()

    def evaluate_variant(
        self,
        text: str,
        mode: str = "ops",
        context: str | None = None,
        context_embedding: List[float] | None = None,
    ) -> Dict[str, Any]:
        """
        Evalúa una variante individual y devuelve el desglose de decisiones.
        """

        reasons: List[str] = []
        scores: Dict[str, Any] = {}

        raw = text or ""
        t = raw.strip()

        mode_min, mode_max = MODE_LENGTH_LIMITS.get(mode, (MIN_LEN, MAX_LEN))
        length_ok = mode_min <= len(t) <= mode_max
        if not length_ok:
            reasons.append(
                f"length_out_of_bounds:{len(t)} not_in[{mode_min},{mode_max}]"
            )
        scores["length_ok"] = length_ok

        lower = t.lower()

        style_basic_ok = True
        for bad in BANNED_SUBSTRINGS:
            if bad in lower:
                style_basic_ok = False
                reasons.append(f"banned_substring:{bad}")
                break

        if style_basic_ok:
            for pref in BANNED_PREFIXES:
                if lower.startswith(pref):
                    style_basic_ok = False
                    reasons.append(f"banned_prefix:{pref}")
                    break

        if CHECKLIST_PATTERN.match(t):
            style_basic_ok = False
            reasons.append("looks_like_generic_checklist")

        if style_basic_ok and "ai" in lower and "you" not in lower and "client" not in lower:
            style_basic_ok = False
            reasons.append("ai_meta_reference")

        scores["style_basic_ok"] = style_basic_ok

        neighbors: List[Tuple[Dict, float]] = self._vs.query(t, k=5)
        max_sim = max((score for _, score in neighbors), default=0.0)

        too_generic = max_sim < MIN_SIM
        too_close = max_sim > MAX_SIM

        if too_generic:
            reasons.append(f"too_generic_sim<{MIN_SIM:.2f}")
        if too_close:
            reasons.append(f"too_close_to_goldset_sim>{MAX_SIM:.2f}")

        scores["similarity"] = float(max_sim)
        scores["too_generic"] = too_generic
        scores["too_close"] = too_close

        confrontational = False
        context_reference_ok = True
        context_similarity: float | None = None
        hook_variety_ok = True
        attacks_author = False
        if mode == "reply":
            confrontation_terms = [
                "you're wrong",
                "you are wrong",
                "this is wrong",
                "let me correct",
            ]
            if lower.startswith("no,") or lower.startswith("no ") or lower.startswith("actually,"):
                confrontational = True
            if any(term in lower for term in confrontation_terms):
                confrontational = True
            if confrontational:
                reasons.append("confrontational_tone")

            attack_patterns = [
                r"\byou(?:'re| are)\s+(?:not|never|wrong|the)",
                r"\byou\s+(?:can't|shouldn't|won't|need to|must|have to)",
                r"\byour\s+(?:fault|problem|mess)",
            ]
            for pattern in attack_patterns:
                if re.search(pattern, lower):
                    attacks_author = True
                    reasons.append("attacks_author")
                    break

            hook_prefix = lower[:40].strip()
            for banned in REPLY_BANNED_HOOKS:
                if hook_prefix.startswith(banned):
                    hook_variety_ok = False
                    reasons.append(f"recycled_hook:{banned}")
                    break

            if context and context_embedding:
                try:
                    reply_embedding = get_embedding(t)
                    context_similarity = _cosine_similarity(
                        reply_embedding, context_embedding
                    )
                    if context_similarity < REPLY_MIN_CONTEXT_SIM:
                        context_reference_ok = False
                        reasons.append(
                            f"context_similarity_low<{REPLY_MIN_CONTEXT_SIM:.2f}"
                        )
                    elif context_similarity > REPLY_MAX_CONTEXT_SIM:
                        reasons.append(
                            f"context_similarity_high>{REPLY_MAX_CONTEXT_SIM:.2f}"
                        )
                except EmbeddingError:
                    context_similarity = None

            if context_similarity is None and context:
                words = {
                    w
                    for w in re.findall(r"[a-z0-9']+", context.lower())
                    if len(w) >= 4
                }
                if words:
                    context_reference_ok = any(word in lower for word in words)
                    if not context_reference_ok:
                        reasons.append("no_context_reference")

        cop_score = ai_cop_score(t)
        ai_suspect = cop_score["suspect"]
        if ai_suspect:
            reasons.append("ai_suspect:" + ",".join(cop_score["reasons"]))
        scores["ai_suspect"] = ai_suspect
        scores["ai_soft_ratio"] = cop_score["soft_ratio"]

        scores["context_similarity"] = context_similarity
        scores["hook_variety_ok"] = hook_variety_ok
        scores["attacks_author"] = attacks_author

        if mode == "ops":
            accepted = (
                length_ok
                and style_basic_ok
                and not too_generic
                and not too_close
                and not ai_suspect
            )
        elif mode == "reply":
            accepted = (
                length_ok
                and style_basic_ok
                and not too_close
                and not confrontational
                and not attacks_author
                and not ai_suspect
                and context_reference_ok
                and hook_variety_ok
            )
        else:
            accepted = length_ok and style_basic_ok and not too_close and not ai_suspect

        return {
            "text": raw,
            "accepted": accepted,
            "reasons": reasons,
            "scores": scores,
        }

    def filter_variants(self, variants: List[str], mode: str = "ops") -> List[Dict[str, Any]]:
        """
        Evalúa una lista de variantes y devuelve solo las aceptadas.
        """

        results: List[Dict[str, Any]] = []
        for variant in variants:
            evaluation = self.evaluate_variant(variant, mode=mode)
            if evaluation["accepted"]:
                results.append(evaluation)
        return results
