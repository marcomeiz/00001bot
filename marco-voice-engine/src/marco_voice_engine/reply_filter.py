"""Filtro previo para decidir si vale la pena responder un tweet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import json
import re

from .config import get_voice_data_dir, get_topics_filename
from .vector_store import VectorStore

MIN_SIMILARITY = 0.34


def _load_topics_text() -> List[str]:
    base_dir = get_voice_data_dir()
    topics_path = base_dir / get_topics_filename()
    if not topics_path.exists():
        return []

    try:
        payload = json.loads(topics_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    topics: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            val = node.strip()
            if val:
                topics.append(val)
            return

        if isinstance(node, dict):
            candidate = (
                node.get("theme")
                or node.get("topic")
                or node.get("idea")
                or node.get("text")
            )
            if isinstance(candidate, str):
                val = candidate.strip()
                if val:
                    topics.append(val)

            for key in ("themes", "topics", "items", "categories"):
                nested = node.get(key)
                if nested:
                    _walk(nested)
            return

        if isinstance(node, list):
            for element in node:
                _walk(element)

    _walk(payload)
    return topics


def _build_keywords() -> List[str]:
    topics = _load_topics_text()
    keywords: List[str] = []
    for text in topics:
        for match in re.findall(r"[a-zA-Z]{4,}", text.lower()):
            if match not in keywords:
                keywords.append(match)
    return keywords


TOPIC_KEYWORDS = _build_keywords()


@dataclass
class FilterResult:
    accepted: bool
    similarity: float
    keyword_hit: bool
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "similarity": self.similarity,
            "keyword_hit": self.keyword_hit,
            "reason": self.reason,
        }


class ReplyFilter:
    """Decide si un tweet es relevante para generar conversación."""

    def __init__(self, min_similarity: float = MIN_SIMILARITY) -> None:
        self._vs = VectorStore.shared()
        self._min_sim = min_similarity
        self._keywords = set(TOPIC_KEYWORDS)

    def evaluate(self, tweet: str) -> FilterResult:
        if not tweet or not tweet.strip():
            return FilterResult(
                accepted=False,
                similarity=0.0,
                keyword_hit=False,
                reason="empty_tweet",
            )

        text = tweet.strip()
        neighbors = self._vs.query(text, k=5)
        similarity = max((score for _, score in neighbors), default=0.0)

        lowered = text.lower()
        keyword_hit = any(keyword in lowered for keyword in self._keywords)

        if similarity >= self._min_sim or keyword_hit:
            return FilterResult(
                accepted=True,
                similarity=float(similarity),
                keyword_hit=keyword_hit,
                reason=None,
            )

        return FilterResult(
            accepted=False,
            similarity=float(similarity),
            keyword_hit=keyword_hit,
            reason="off_topic",
        )
