"""
FastAPI wrapper for marco-voice-engine with Supabase embedding cache.
"""

from __future__ import annotations

import os
import json
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from src.marco_voice_engine.config import get_voice_data_dir
from src.marco_voice_engine.generator import Generator
from src.marco_voice_engine.judge import Judge
from src.marco_voice_engine.embeddings import get_embedding
from src.marco_voice_engine.vector_store import VectorStore


app = FastAPI(
    title="Marco Voice Engine API",
    description="AI-powered tweet generation with Supabase embedding cache",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)

# Supabase client (opcional)
_SUPA_URL = os.environ.get("SUPABASE_URL", "")
_SUPA_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
supabase: Client | None = None
if _SUPA_URL and _SUPA_KEY:
    supabase = create_client(_SUPA_URL, _SUPA_KEY)


class GenerateRequest(BaseModel):
    mode: Literal["ops", "chaos"]
    min_diff: Optional[float] = None
    prompt: Optional[str] = None  # Custom prompt parameter


class Variant(BaseModel):
    text: str
    score: float


class GenerateResponse(BaseModel):
    topic: str
    variants: List[Variant]


def load_topics() -> List[Dict[str, Any]]:
    """Load all topics from topics.json."""
    data_dir = Path(get_voice_data_dir())
    topics_path = data_dir / "topics.json"

    if not topics_path.exists():
        raise FileNotFoundError(f"topics.json not found at {topics_path}")

    with open(topics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_themes = []
    for category in data.get("categories", []):
        for theme in category.get("themes", []):
            all_themes.append({
                "category": category["category"],
                "theme": theme["theme"],
                "id": theme["id"],
            })

    return all_themes


def select_random_topic() -> Dict[str, Any]:
    """Select a random topic from topics.json."""
    topics = load_topics()
    if not topics:
        raise ValueError("No topics available in topics.json")
    return random.choice(topics)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "marco-voice-engine-api",
        "version": "2.0.0",
        "embedding_cache": "supabase"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/admin/precompute")
async def precompute_embeddings():
    """
    Admin endpoint: Pre-compute embeddings for goldset and store in Supabase.
    Run this ONCE after deploy or when goldset changes.
    """
    if supabase is None:
        raise HTTPException(status_code=400, detail="Supabase not configured")
    print("[PRECOMPUTE] Starting embedding pre-computation...")
    
    def do_precompute():
        from src.marco_voice_engine.style_retriever import StyleRetriever
        from src.marco_voice_engine.embeddings import get_embedding
        
        try:
            # Load goldset
            data_dir = Path(get_voice_data_dir())
            goldset_path = data_dir / os.environ.get("GOLDSET_FILENAME", "dataset.json")
            
            with open(goldset_path, "r", encoding="utf-8") as f:
                goldset_data = json.load(f)
                # Extract tweets array from dataset.json structure
                if isinstance(goldset_data, dict) and "tweets" in goldset_data:
                    goldset_lines = goldset_data["tweets"]
                elif isinstance(goldset_data, list):
                    goldset_lines = goldset_data
                else:
                    goldset_lines = [goldset_data]
            
            print(f"[PRECOMPUTE] Loaded {len(goldset_lines)} goldset entries")
            
            # Check existing embeddings
            existing = supabase.table("goldset_embeddings").select("text").execute()
            existing_texts = {row["text"] for row in existing.data}
            
            new_entries = [entry for entry in goldset_lines if entry.get("text") not in existing_texts]
            
            if not new_entries:
                print("[PRECOMPUTE] All embeddings already cached!")
                return {"message": "All embeddings already cached", "total": len(goldset_lines)}
            
            print(f"[PRECOMPUTE] Computing embeddings for {len(new_entries)} new entries...")
            
            # Compute embeddings
            texts_to_embed = [entry.get("text", "") for entry in new_entries]
            embeddings = [get_embedding(text) for text in texts_to_embed]
            
            # Store in Supabase
            records = []
            for entry, embedding in zip(new_entries, embeddings):
                records.append({
                    "text": entry.get("text", ""),
            "embedding": embedding,
                    "metadata": entry
                })
            
            # Batch insert
            print(f"[PRECOMPUTE] Inserting {len(records)} records into Supabase...")
            supabase.table("goldset_embeddings").insert(records).execute()
            
            print("[PRECOMPUTE] Done!")
            return {
                "message": "Embeddings computed and cached",
                "new_entries": len(new_entries),
                "total_cached": len(goldset_lines)
            }
            
        except Exception as e:
            print(f"[PRECOMPUTE] ERROR: {e}")
            raise
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, do_precompute)
    return result


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Genera variantes con IA y devuelve solo las que pasan filtros de calidad.
    Usa el índice por defecto (`VectorStore.shared`) como fallback si no hay cache en Supabase.
    """

    LAST_TOPIC_META: Dict[str, Any] | None = getattr(generate, "_last_topic_meta", None)
    def run_generation() -> Dict[str, Any]:
        try:
            if request.min_diff is not None:
                min_diff = request.min_diff
            else:
                raw_md = os.environ.get("TOPIC_MIN_DIFF", "0.3")
                try:
                    min_diff = float(raw_md)
                except Exception:
                    min_diff = 0.3
            if min_diff < 0.0:
                min_diff = 0.0
            if min_diff > 1.0:
                min_diff = 1.0
            if LAST_TOPIC_META is None:
                topic_data = select_random_topic()
            else:
                topic_data = select_diverse_topic(LAST_TOPIC_META, min_diff)
            topic_text = topic_data["theme"]

            # Log if custom prompt is being used
            if request.prompt:
                print(f"[MARCO] Using custom prompt for {request.mode} mode: {request.prompt[:100]}...")
            else:
                print(f"[MARCO] Using default system prompt for {request.mode} mode")

            generator = Generator(mode=request.mode)
            variants_text = generator.generate_variants(topic_text, n_expected=2, custom_prompt=request.prompt)

            judge = Judge()
            accepted_records = judge.filter_variants(variants_text, mode=request.mode)

            variants_payload: List[Dict[str, Any]] = []
            for rec in accepted_records[:2]:
                raw_sim = rec.get("scores", {}).get("similarity", 0.0)
                try:
                    print("[API] similarity_raw_type=", type(raw_sim), "value_sample=", str(raw_sim)[:64])
                except Exception:
                    pass
                score = raw_sim if isinstance(raw_sim, (int, float)) else 0.0
                variants_payload.append({
                    "text": rec.get("text", ""),
                    "score": float(score),
                })

            return {
                "topic": topic_text,
                "variants": variants_payload,
            }
        except Exception as e:
            raise RuntimeError(f"generation_failed: {e}")
    # fallback eliminado: si falla el proveedor o no hay variantes válidas, se devuelve error

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_generation)
        variants = [Variant(**v) for v in result["variants"]]
        if not variants:
            raise HTTPException(status_code=502, detail="No variants passed quality filters")
        setattr(generate, "_last_topic_meta", {"theme": result["topic"]})
        return GenerateResponse(topic=result["topic"], variants=variants)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
@app.get("/metrics/vector-store")
async def vector_store_metrics():
    if supabase is not None:
        try:
            table = os.environ.get("SUPABASE_EMBED_TABLE", "goldset_embeddings")
            data = supabase.table(table).select("text").execute()
            return {"items": len(data.data or []), "source": "supabase"}
        except Exception:
            pass
    return {"items": 0, "source": "unknown"}
def load_topic_embeddings() -> List[Tuple[Dict[str, Any], List[float]]]:
    topics = load_topics()
    if supabase is not None:
        try:
            table = os.environ.get("SUPABASE_TOPIC_TABLE", "topics_embeddings")
            data = supabase.table(table).select("id, theme, embedding").execute()
            rows = data.data or []
            if rows:
                emap: Dict[str, List[float]] = {}
                for r in rows:
                    key = str(r.get("id") or r.get("theme") or "")
                    if key:
                        emap[key] = r.get("embedding")
                out: List[Tuple[Dict[str, Any], List[float]]] = []
                for t in topics:
                    key = str(t.get("id") or t.get("theme") or "")
                    emb = emap.get(key)
                    if emb is None:
                        continue  # no calcular al vuelo
                    out.append((t, emb))
                return out
        except Exception:
            pass
    # sin embeddings precomputados, devolver vacío (se usará fallback por categoría)
    return []

def select_diverse_topic(last_meta: Optional[Dict[str, Any]], min_diff: float) -> Dict[str, Any]:
    pairs = load_topic_embeddings()
    if pairs:
        # usar embeddings precomputados
        if last_meta is None:
            return random.choice([p[0] for p in pairs])
        import numpy as np
        # intentar recuperar embedding del último topic
        last_emb = None
        try:
            table = os.environ.get("SUPABASE_TOPIC_TABLE", "topics_embeddings")
            res = supabase.table(table).select("embedding").eq("id", last_meta.get("id")).limit(1).execute()
            if res.data:
                last_emb = res.data[0].get("embedding")
        except Exception:
            pass
        if last_emb is None:
            return random.choice([p[0] for p in pairs])
        v_last = np.array(last_emb, dtype="float32")
        v_last = v_last / (np.linalg.norm(v_last) + 1e-9)
        scored: List[Tuple[Dict[str, Any], float]] = []
        for t, emb in pairs:
            v = np.array(emb, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            sim = float(v_last.dot(v))
            diff = 1.0 - max(min(sim, 1.0), -1.0)
            scored.append((t, diff))
        scored.sort(key=lambda x: x[1], reverse=True)
        for t, diff in scored:
            if diff >= min_diff:
                return t
        return scored[0][0]
    # sin embeddings: elegir por categoría distinta
    topics = load_topics()
    if not topics:
        raise ValueError("No topics available")
    if last_meta is None:
        return random.choice(topics)
    last_cat = last_meta.get("category")
    diff_cat = [t for t in topics if t.get("category") != last_cat]
    return random.choice(diff_cat or topics)
