"""Selector de ejemplos estilísticos basados en el goldset."""

from __future__ import annotations

from typing import Dict, List, Optional
import random

from .goldset_loader import load_goldset
from .vector_store import VectorStore

DEFAULT_K = 12


class StyleRetriever:
    """
    Selecciona ejemplos del goldset para enseñar ESTILO al generador.
    No decide temas, solo cómo suena Marco.
    """

    def __init__(self) -> None:
        self._goldset: List[Dict] = load_goldset()
        if not self._goldset:
            raise RuntimeError("Goldset is empty in StyleRetriever.")
        self._vs = VectorStore.shared()

    def select_examples(
        self,
        idea: str,
        k: int = DEFAULT_K,
        tone: Optional[str] = None,
        formato: str = "tweet",
        style: Optional[str] = None,
    ) -> List[Dict]:
        """
        Devuelve una lista de ejemplos (dicts) para usar en el prompt.
        Reglas:
        - Prioriza semanticamente cercanos a la idea, PERO
        - Filtra por formato,
        - Filtra opcionalmente por tone,
        - Introduce diversidad (no siempre las mismas estructuras),
        - Nunca devuelve duplicados.
        """

        if k <= 0:
            return []

        candidates = self._vs.query(idea, k=k * 3)
        items = [item for (item, _score) in candidates]

        items = [it for it in items if it.get("formato") == formato]

        preferred_styles: List[str] | None = None
        if style:
            if style == "ops":
                preferred_styles = ["ops", "both"]
            elif style == "chaos":
                preferred_styles = ["chaos", "both"]
            elif style == "reply":
                preferred_styles = ["ops", "both", "neutral", "chaos"]
            elif style == "neutral":
                preferred_styles = ["neutral", "ops", "chaos", "both"]

        if tone:
            filtered = [it for it in items if it.get("tone") == tone]
            if filtered:
                items = filtered

        if preferred_styles:
            filtered = [it for it in items if it.get("style") in preferred_styles]
            if filtered:
                items = filtered

        if len(items) < k:
            extra = [it for it in self._goldset if it.get("formato") == formato]
            seen_ids = {it["id"] for it in items if "id" in it}
            extra = [it for it in extra if it.get("id") not in seen_ids]

            prioritized: List[Dict] = []
            if preferred_styles:
                prioritized = [it for it in extra if it.get("style") in preferred_styles]
                extra = [it for it in extra if it.get("style") not in preferred_styles]

            random.shuffle(prioritized)
            random.shuffle(extra)
            needed = max(0, k - len(items))
            replenishment = prioritized + extra
            items.extend(replenishment[:needed])

        random.shuffle(items)

        return items[:k]
