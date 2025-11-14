"""Vector store basado en FAISS para consultar el goldset."""

from __future__ import annotations

from typing import List, Tuple
import json

import faiss
import numpy as np
import os

from .goldset_loader import load_goldset
from .embeddings import get_embedding
from supabase import create_client


class VectorStore:
    _shared_instance: "VectorStore" | None = None

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.items: List[dict] = []
        self.source: str = "unknown"

    def build(self) -> None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_ANON_KEY", "")
        table = os.getenv("SUPABASE_EMBED_TABLE", "goldset_embeddings")

        # Si Supabase está configurado, SOLO cargar desde Supabase; nunca recalcular desde dataset
        if url and key:
            client = create_client(url, key)
            data = client.table(table).select("text, embedding, metadata").execute()
            rows = data.data or []
            if not rows:
                raise RuntimeError("Supabase embeddings table is empty. Run /admin/precompute or populate 'goldset_embeddings'.")
            vectors = []
            for r in rows:
                emb = r.get("embedding")
                if isinstance(emb, str):
                    try:
                        emb = json.loads(emb)
                    except Exception:
                        raise RuntimeError("Invalid embedding format in Supabase: expected list, got string")
                vectors.append(np.array(emb, dtype="float32"))
            self.items = [r.get("metadata") or {"text": r.get("text", "")} for r in rows]
            mat = np.stack(vectors).astype("float32")
            dim = mat.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(mat)
            self.index.add(mat)
            print(f"[VectorStore] Loaded {len(rows)} embeddings from Supabase table '{table}' (dim={dim}).")
            self.source = "supabase"
            return

        # Fallback SOLO si no hay Supabase configurado: construir desde dataset
        goldset = load_goldset()
        if not goldset:
            raise RuntimeError("Goldset is empty. Cannot build index.")

        vectors = []
        self.items = []
        for item in goldset:
            emb = get_embedding(item["text"])
            vectors.append(emb)
            self.items.append(item)

        mat = np.array(vectors, dtype="float32")
        if mat.ndim != 2:
            raise RuntimeError("Embeddings must form a 2D matrix.")

        dim = mat.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(mat)
        self.index.add(mat)
        print(f"[VectorStore] Built index from goldset: {len(self.items)} items (dim={dim}).")
        self.source = "goldset"

    def query(self, text: str, k: int = 10) -> List[Tuple[dict, float]]:
        """
        Devuelve top-k items similares (item, score).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        if not self.items or k <= 0:
            return []

        top_k = min(k, len(self.items))
        vec = np.array([get_embedding(text)], dtype="float32")
        faiss.normalize_L2(vec)
        scores, idxs = self.index.search(vec, top_k)

        results: List[Tuple[dict, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.items[idx], float(score)))

        return results

    @classmethod
    def shared(cls) -> "VectorStore":
        """
        Devuelve una instancia singleton para evitar reconstruir el índice.
        """

        if cls._shared_instance is None:
            instance = cls()
            instance.build()
            cls._shared_instance = instance
        return cls._shared_instance
