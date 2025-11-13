"""
FastAPI wrapper for marco-voice-engine with Supabase embedding cache.
"""

from __future__ import annotations

import os
import json
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Literal
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from src.marco_voice_engine.config import get_voice_data_dir


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

# Supabase client
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL", ""),
    os.environ.get("SUPABASE_ANON_KEY", "")
)


class GenerateRequest(BaseModel):
    mode: Literal["ops", "chaos"]


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
    Generate AI-powered tweet variants using cached embeddings from Supabase.
    """
    print("=== GENERATE START ===")
    
    def run_generation():
        from src.marco_voice_engine.generator import Generator
        from src.marco_voice_engine.judge import Judge
        import numpy as np
        
        try:
            print("1. Loading topic...")
            topic_data = select_random_topic()
            topic_text = topic_data["theme"]
            print(f"2. Topic selected: {topic_text[:50]}...")

            print("3. Fetching cached embeddings from Supabase...")
            cached = supabase.table("goldset_embeddings").select("text, embedding, metadata").execute()
            
            if not cached.data:
                raise ValueError("No embeddings in cache. Run /admin/precompute first!")
            
            print(f"4. Loaded {len(cached.data)} cached embeddings")
            
            # Reconstruct goldset with embeddings
            goldset_entries = []
            goldset_embeddings = []
            
            for row in cached.data:
                goldset_entries.append(row["metadata"])
                goldset_embeddings.append(np.array(row["embedding"]))
            
            goldset_embeddings = np.array(goldset_embeddings)
            
            print("5. Generating variants...")
            generator = Generator(mode=request.mode)
            variants_text = generator.generate_variants(topic_text, n_expected=2)
            
            print("6. Judging variants...")
            judge = Judge(goldset_embeddings=goldset_embeddings)
            result = judge.judge(variants_text, mode=request.mode)
            
            print("7. Generation completed")
            return result, topic_text
            
        except Exception as e:
            print(f"ERROR in generation: {e}")
            raise
    
    try:
        loop = asyncio.get_event_loop()
        result, topic_text = await loop.run_in_executor(executor, run_generation)
        
        accepted = result.get("accepted", [])
        
        if not accepted:
            raise HTTPException(
                status_code=500,
                detail="No variants passed quality filters. Try again."
            )
        
        variants = []
        for variant in accepted[:2]:
            variants.append(
                Variant(
                    text=variant.get("text", ""),
                    score=variant.get("similarity", 0.0),
                )
            )
        
        if len(variants) == 1:
            rejected = result.get("rejected", [])
            if rejected:
                best_rejected = max(rejected, key=lambda x: x.get("similarity", 0.0))
                variants.append(
                    Variant(
                        text=best_rejected.get("text", ""),
                        score=best_rejected.get("similarity", 0.0),
                    )
                )
        
        print("8. Returning response")
        return GenerateResponse(
            topic=topic_text,
            variants=variants,
        )
    
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    except ValueError as e:
        print(f"ERROR: Value error - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"ERROR: Exception - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
