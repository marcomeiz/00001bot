"""Centraliza la configuración del proyecto sin valores hardcodeados."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv_path = _PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


@dataclass(frozen=True)
class Settings:
    """Agrupa claves y rutas. TODOS los valores deben venir de entorno."""

    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    model_name: Optional[str] = os.getenv("VOICE_MODEL_NAME")
    data_dir: Optional[Path] = (
        Path(os.getenv("VOICE_DATA_DIR")) if os.getenv("VOICE_DATA_DIR") else None
    )
    goldset_filename: Optional[str] = os.getenv("GOLDSET_FILENAME")


settings = Settings()

# TODO: Validar los campos obligatorios cuando se implemente la lógica real.


def get_openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def get_embedding_model() -> str | None:
    return os.getenv("EMBEDDING_MODEL_NAME")


def get_generation_model() -> str | None:
    return os.getenv("GENERATION_MODEL_NAME")


def get_openrouter_chat_endpoint() -> str:
    return os.getenv(
        "OPENROUTER_CHAT_ENDPOINT",
        "https://openrouter.ai/api/v1/chat/completions",
    )


def get_generation_model_primary() -> str | None:
    return os.getenv("GENERATION_MODEL_PRIMARY")


def get_generation_model_secondary() -> str | None:
    return os.getenv("GENERATION_MODEL_SECONDARY")


def _autodetect_dataset() -> Path | None:
    """
    Busca dataset.json en la raíz del repo para usarlo como fallback.
    """

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "dataset.json"
    if candidate.exists():
        return candidate
    return None


def get_voice_data_dir() -> Path:
    value = os.getenv("VOICE_DATA_DIR", "")
    if value:
        return Path(value)

    detected = _autodetect_dataset()
    if detected:
        return detected.parent

    return Path(__file__).resolve().parents[2] / "data"


def get_goldset_filename() -> str:
    value = os.getenv("GOLDSET_FILENAME")
    if value:
        return value

    detected = _autodetect_dataset()
    if detected:
        return detected.name

    return "goldset.example.jsonl"


def get_topics_filename() -> str:
    return os.getenv("TOPICS_FILENAME", "topics.json")
