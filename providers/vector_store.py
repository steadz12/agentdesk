"""
providers/vector_store.py — Swappable Vector Store Backends
──────────────────────────────────────────────────────────────────────
Swap the vector store by changing one env var:

    VECTOR_STORE=memory    # default — zero dependencies, great for demos
    VECTOR_STORE=faiss     # fast local search (pip install agentdesk[faiss])
    VECTOR_STORE=pinecone  # managed cloud (pip install agentdesk[pinecone])

All stores expose the same interface:
    store.add(texts, metadatas)   — index documents
    store.search(query, top_k)    — retrieve top-k results as (text, score, meta)

Adding a new store:
    1. Subclass VectorStore
    2. Implement add() and search()
    3. Add a branch in get_vector_store()
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Base class for all vector store backends."""

    @abstractmethod
    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...

    def __len__(self) -> int:
        return 0


# ── In-memory TF-IDF (default, zero deps) ─────────────────────────────────────

class InMemoryVectorStore(VectorStore):
    """
    Zero-dependency TF-IDF vector store.
    Fast enough for thousands of documents. No GPU or cloud needed.

    Upgrade path: swap for FAISSVectorStore or PineconeVectorStore
    when you need semantic search or scale.
    """

    def __init__(self):
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self._texts.extend(texts)
        self._metas.extend(metadatas)
        self._rebuild_index()

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self._texts:
            return []
        q_vec = self._embed(query)
        scores = [self._cosine(q_vec, v) for v in self._vectors]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(text=self._texts[i], score=s, metadata=self._metas[i])
            for i, s in ranked[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._texts)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _rebuild_index(self) -> None:
        df: Counter = Counter()
        all_tokens = [self._tokenize(t) for t in self._texts]
        for tokens in all_tokens:
            df.update(set(tokens))
        N = len(self._texts)
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}
        self._vectors = [self._tfidf(tok) for tok in all_tokens]

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        tf: Counter = Counter(tokens)
        n = len(tokens) or 1
        return {t: (tf[t] / n) * self._idf.get(t, 0) for t in tf}

    def _embed(self, text: str) -> dict[str, float]:
        return self._tfidf(self._tokenize(text))

    def _cosine(self, a: dict, b: dict) -> float:
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
        na = math.sqrt(sum(v ** 2 for v in a.values()))
        nb = math.sqrt(sum(v ** 2 for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


# ── FAISS (optional — pip install agentdesk[faiss]) ───────────────────────────

class FAISSVectorStore(VectorStore):
    """
    Local semantic vector search using FAISS + simple bag-of-words embeddings.
    For production, replace _embed() with a real embedding model
    (Voyage AI, OpenAI text-embedding-3-small, etc.)

    Install: pip install agentdesk[faiss]
    """

    def __init__(self, dim: int = 512):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("Run: pip install agentdesk[faiss]")
        self._faiss = faiss
        self._np = np
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)   # inner product = cosine if normalised
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vocab: dict[str, int] = {}

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for t in texts:
            for w in t.lower().split():
                if w not in self._vocab and len(self._vocab) < self._dim:
                    self._vocab[w] = len(self._vocab)
        vecs = self._np.array([self._embed(t) for t in texts], dtype="float32")
        self._faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._texts.extend(texts)
        self._metas.extend(metadatas)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self._texts:
            return []
        q = self._np.array([self._embed(query)], dtype="float32")
        self._faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(top_k, len(self._texts)))
        return [
            SearchResult(text=self._texts[i], score=float(s), metadata=self._metas[i])
            for s, i in zip(scores[0], idxs[0]) if i >= 0
        ]

    def __len__(self) -> int:
        return len(self._texts)

    def _embed(self, text: str) -> "np.ndarray":
        vec = self._np.zeros(self._dim, dtype="float32")
        for w in text.lower().split():
            idx = self._vocab.get(w)
            if idx is not None:
                vec[idx] += 1.0
        return vec


# ── Pinecone (optional — pip install agentdesk[pinecone]) ─────────────────────

class PineconeVectorStore(VectorStore):
    """
    Managed cloud vector search via Pinecone.
    Required env: PINECONE_API_KEY, PINECONE_INDEX

    NOTE: You must supply a real embedding function.
    Set PINECONE_EMBED_MODEL or replace _embed() with your own.

    Install: pip install agentdesk[pinecone]
    """

    def __init__(self):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("Run: pip install agentdesk[pinecone]")
        api_key = os.environ["PINECONE_API_KEY"]
        index_name = os.environ["PINECONE_INDEX"]
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._counter = 0

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        vectors = []
        for text, meta in zip(texts, metadatas):
            vec_id = f"doc-{self._counter}"
            self._counter += 1
            vectors.append({
                "id": vec_id,
                "values": self._embed(text),
                "metadata": {**meta, "text": text},
            })
        self._index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        results = self._index.query(vector=self._embed(query), top_k=top_k, include_metadata=True)
        return [
            SearchResult(
                text=m.metadata.get("text", ""),
                score=m.score,
                metadata={k: v for k, v in m.metadata.items() if k != "text"},
            )
            for m in results.matches
        ]

    def _embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Replace _embed() in PineconeVectorStore with a real embedding function, e.g.:\n"
            "  import voyageai; vo = voyageai.Client()\n"
            "  return vo.embed([text], model='voyage-3').embeddings[0]"
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def get_vector_store() -> VectorStore:
    """
    Return the configured vector store.
    Set VECTOR_STORE env var to switch backends.

    Supported values: "memory" (default), "faiss", "pinecone"
    """
    store = os.getenv("VECTOR_STORE", "memory").lower()
    if store == "memory":
        return InMemoryVectorStore()
    if store == "faiss":
        return FAISSVectorStore()
    if store == "pinecone":
        return PineconeVectorStore()
    raise ValueError(
        f"Unknown VECTOR_STORE='{store}'. Supported: memory, faiss, pinecone. "
        "See providers/vector_store.py to add your own."
    )
