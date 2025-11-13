"""Utilidades para leer el conjunto de ejemplos en formato JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from .config import get_goldset_filename, get_voice_data_dir

JsonLike = Mapping[str, Any]
REQUIRED_FIELDS = ("id", "text", "tone", "formato", "topic")
ALLOWED_STYLES = {"ops", "chaos", "both", "neutral"}


def load_goldset(path: str | Path | None = None) -> List[JsonLike]:
    """Carga un JSON o JSONL y devuelve una lista de dicts válidos."""

    target_path = Path(path) if path else _default_goldset_path()
    text = target_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    entries: List[JsonLike] = []
    candidates: List[Any] = []
    expected_total: int | None = None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict) and "tweets" in parsed:
            candidates = parsed["tweets"]
            total = parsed.get("total_tweets")
            if isinstance(total, int):
                expected_total = total
        else:
            candidates = [parsed]
    except json.JSONDecodeError:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))

    for record in candidates:
        normalized = _normalize_entry(record)
        if normalized and _is_valid_goldset_entry(normalized):
            entries.append(normalized)

    if expected_total is not None and len(entries) != expected_total:
        raise ValueError(
            f"Goldset at {target_path} expected {expected_total} entries, "
            f"found {len(entries)}."
        )

    return entries


def _default_goldset_path() -> Path:
    data_dir = get_voice_data_dir()
    filename = get_goldset_filename()
    return data_dir / filename


def iter_goldset(path: str | Path | None = None) -> Iterable[JsonLike]:
    """Itera sobre el goldset independientemente del formato (JSON/JSONL)."""

    for record in load_goldset(path):
        yield record


def _is_valid_goldset_entry(entry: Mapping[str, Any]) -> bool:
    """Verifica mínimos obligatorios del goldset."""

    for field in REQUIRED_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _normalize_entry(entry: Any) -> Dict[str, Any] | None:
    """Limpia strings/tipos sin rellenar campos faltantes obligatorios."""

    if not isinstance(entry, dict):
        return None

    normalized = dict(entry)
    for field in REQUIRED_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = value.replace("\r", " ").strip()

    style_val = normalized.get("style")
    if isinstance(style_val, str):
        style_val = style_val.strip().lower()
    if style_val not in ALLOWED_STYLES:
        normalized["style"] = "neutral"
    else:
        normalized["style"] = style_val

    return normalized
