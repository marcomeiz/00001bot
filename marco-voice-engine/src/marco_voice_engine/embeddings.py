"""Interfaz con OpenRouter para generar embeddings."""

from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import time
import hashlib
from pathlib import Path

import httpx

from .config import (
    get_embedding_model,
    get_openrouter_api_key,
    get_voice_data_dir,
    get_goldset_filename,
)

EMBED_DIM = 3072  # TODO: sincronizar con modelo real

_CACHE_DATA: Dict[str, Any] = {"meta": None, "vectors": {}}
_CACHE_PATH = Path(
    os.getenv(
        "EMBEDDING_CACHE_PATH",
        get_voice_data_dir() / ".cache" / "embeddings.json",
    )
)
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
if _CACHE_PATH.exists():
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
            if isinstance(payload, dict):
                _CACHE_DATA.update(
                    {
                        "meta": payload.get("meta"),
                        "vectors": payload.get("vectors", {}),
                    }
                )
    except Exception:
        _CACHE_DATA = {"meta": None, "vectors": {}}


class EmbeddingError(Exception):
    """Errores derivados de la obtención de embeddings."""


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_mtime() -> str:
    dataset_path = get_voice_data_dir() / get_goldset_filename()
    if dataset_path.exists():
        return str(dataset_path.stat().st_mtime_ns)
    return "missing"


def _current_meta(model: str) -> Dict[str, str]:
    return {
        "model": model,
        "dataset_mtime": _dataset_mtime(),
    }


def _get_cached_vector(key: str, meta: Dict[str, str]) -> List[float] | None:
    stored_meta = _CACHE_DATA.get("meta")
    if stored_meta != meta:
        _CACHE_DATA["meta"] = meta
        _CACHE_DATA["vectors"] = {}
        return None
    vec = _CACHE_DATA["vectors"].get(key)
    if isinstance(vec, list):
        return vec
    return None


def _save_cache() -> None:
    try:
        with _CACHE_PATH.open("w", encoding="utf-8") as cache_file:
            json.dump(_CACHE_DATA, cache_file)
    except Exception:
        pass


def _store_cached_vector(key: str, vec: List[float], meta: Dict[str, str]) -> None:
    if _CACHE_DATA.get("meta") != meta:
        _CACHE_DATA["meta"] = meta
        _CACHE_DATA["vectors"] = {}
    _CACHE_DATA["vectors"][key] = vec
    _save_cache()


def get_embedding(text: str) -> List[float]:
    """
    Devuelve el embedding del texto usando el modelo configurado en OpenRouter.
    Debe ser determinista para el mismo input.
    """
    api_key = get_openrouter_api_key()
    model = get_embedding_model()

    if not api_key or not model:
        raise EmbeddingError("Missing OpenRouter API key or embedding model config.")

    meta = _current_meta(model)
    cache_key = _cache_key(text)
    cached = _get_cached_vector(cache_key, meta)
    if cached is not None:
        return cached

    endpoint = os.environ.get(
        "OPENROUTER_EMBEDDINGS_ENDPOINT",
        "https://openrouter.ai/api/v1/embeddings",
    )

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": text,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            try:
                vec = data["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as err:
                raise EmbeddingError(f"Invalid embedding payload: {data}") from err
            break
        except Exception as err:  # pragma: no cover - network stub
            last_err = err
            if attempt == 2:
                raise EmbeddingError(f"Error getting embedding: {err}") from err
            time.sleep(1 + attempt)
    else:  # pragma: no cover
        raise EmbeddingError(f"Error getting embedding: {last_err}")

    if not isinstance(vec, list) or not vec:
        raise EmbeddingError("Invalid embedding format from provider.")
    _store_cached_vector(cache_key, vec, meta)
    return vec
