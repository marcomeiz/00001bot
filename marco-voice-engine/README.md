# Marco Voice Engine

Plantilla inicial para un motor de generación y evaluación de variantes de voz/marca.

## Estado actual
- Python 3.11+
- Estructura `src/` lista para expansión modular
- Configuración centralizada en `src/config.py`

## Estructura
```
marco-voice-engine/
  README.md
  pyproject.toml
  src/
    __init__.py
    config.py
    goldset_loader.py
    embeddings.py
    vector_store.py
    generator.py
    judge.py
  data/
    goldset.example.jsonl
```

## Configuración
Define las siguientes variables de entorno antes de añadir lógica real:
- `OPENROUTER_API_KEY`
- `VOICE_DATA_DIR`
- `GOLDSET_FILENAME`
- `TOPICS_FILENAME`
- `EMBEDDING_MODEL_NAME`
- `GENERATION_MODEL_PRIMARY`
- `GENERATION_MODEL_SECONDARY` (solo si quieres un sparring)

## V1_STABLE
- **Goldset**: `dataset.json` (ubicado en `VOICE_DATA_DIR`). `topics.json` es solo una bolsa de ideas, nunca se usa para aprender estilo.
- Cada tweet en `dataset.json` trae `style` (`ops`, `chaos`, `both`, `neutral`). Se usa para elegir qué cara (operador vs feral) alimentar al generador.
- **Embeddings**: `EMBEDDING_MODEL_NAME` (ej. `openai/text-embedding-3-large`). El cache reside en `<VOICE_DATA_DIR>/.cache/embeddings.json`; si cambias el dataset o el modelo, borra ese archivo.
- **Modelos**: Solo se invoca `GENERATION_MODEL_PRIMARY`. Deja `GENERATION_MODEL_SECONDARY` sin definir para desactivar el sparring.
- **Judge**: Usa `MIN_SIM=0.60` y `MAX_SIM=0.97`, más un bloque de frases prohibidas; ninguna variante supera el filtro si viola longitudes, similitud o reglas anti-coach (sin emojis, sin hashtags, sin frases de coach barato).
- **Seguridad**: No hay publicación automática. Todo el output sale por `stdout` o como retorno de funciones para integrar en tu propio flujo.

## Configuración
1. Copia `.env.example` a `.env` en la raíz de este repo.
2. Completa tu `OPENROUTER_API_KEY` y ajusta rutas/modelos si hace falta.
3. Ejecuta `python -m marco_voice_engine "You worked 60 hours last week and nothing moved."` para verificar.

## Bot privado de Telegram
1. Añade en `.env`:
   - `TELEGRAM_BOT_TOKEN`
   - `ALLOWED_USER_IDS` (IDs autorizados separados por comas)
2. Instala dependencias (`pip install -e .`).
3. Ejecuta `python -m marco_voice_engine.telegram_bot` y pulsa OPS/CHAOS para recibir propuestas basadas en topics aleatorios de tu lista.

## CLI rápido
- `python -m marco_voice_engine "idea"` → modo `ops`.
- `python -m marco_voice_engine chaos "idea"` → modo `chaos`.

## Instalación
```bash
pip install -e .
```

## Próximos pasos
1. Implementar `embeddings.get_embedding`
2. Construir `vector_store` real (FAISS u otro)
3. Conectar `generator` y `judge` a modelos reales

## Registro de cambios
- 2025-11-11 · Autor: Codex · Crear esqueleto inicial. Justificación: necesidad de base limpia y modular para las siguientes instrucciones.
