from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.config import INDEXES_DIR, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.models import ChunkRecord, SearchResult

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorStore:
    def __init__(self, base_dir: Path = INDEXES_DIR) -> None:
        self.base_dir = base_dir

    def _path(self, paper_id: str) -> Path:
        return self.base_dir / f"{paper_id}.json"

    def index_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)
        indexed: list[ChunkRecord] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            indexed.append(chunk.model_copy(update={"embedding": embedding}))
        self.save(indexed[0].paper_id, indexed)
        return indexed

    def save(self, paper_id: str, chunks: list[ChunkRecord]) -> None:
        payload = [c.model_dump() for c in chunks]
        self._path(paper_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, paper_id: str) -> list[ChunkRecord]:
        path = self._path(paper_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ChunkRecord.model_validate(item) for item in data]

    def search(self, paper_id: str, query: str, top_k: int = 4) -> list[SearchResult]:
        return self.search_many(paper_id, [query], top_k=top_k)

    def search_many(self, paper_id: str, queries: list[str], top_k: int = 4) -> list[SearchResult]:
        """Multi-query search with a single embedding request and one disk load.

        Each chunk is scored by its best similarity across all queries, so the
        caller gets the top_k most relevant chunks overall (deduplicated).
        """
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []

        chunks = self.load(paper_id)
        if not chunks:
            return []

        query_vecs = [np.array(v, dtype=float) for v in self._embed_texts(queries)]
        scored: list[SearchResult] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            vec = np.array(chunk.embedding, dtype=float)
            best = max(_cosine(qv, vec) for qv in query_vecs)
            scored.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=best,
                    section=chunk.section,
                    page=chunk.page,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                logger.warning("OpenAI embeddings failed, using fallback: %s", exc)

        return [self._fallback_embedding(text) for text in texts]

    @staticmethod
    def _fallback_embedding(text: str, dim: int = 256) -> list[float]:
        vec = np.zeros(dim, dtype=float)
        tokens = text.lower().split()
        for token in tokens:
            idx = hash(token) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        return vec.tolist()
