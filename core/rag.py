"""
core/rag.py — RAG Engine (provider-agnostic)
"""
import os
from providers.vector_store import VectorStore, SearchResult

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Ingest documents -> chunk -> index via VectorStore -> retrieve on demand.
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def ingest_directory(self, path: str) -> int:
        """Index all .txt files from a directory."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return 0
        total = 0
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".txt"):
                fpath = os.path.join(path, fname)
                with open(fpath) as f:
                    text = f.read()
                total += self.ingest_text(text, source=fname)
        return total

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Chunk a string and add to the vector store."""
        chunks = self._split(text)
        metas  = [{"source": source, "chunk": i} for i in range(len(chunks))]
        self.store.add(chunks, metas)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[SearchResult]:
        return self.store.search(query, top_k=top_k)

    def format_context(self, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant context found in knowledge base."
        parts = [f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.text}" for r in results]
        return "\n\n---\n\n".join(parts)

    def _split(self, text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
