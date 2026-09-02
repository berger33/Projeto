from __future__ import annotations

import math

from .domain import Chunk, RetrievedChunk
from .embeddings import EmbeddingProvider


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vetores com dimensões diferentes.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider):
        if not chunks:
            raise ValueError("O índice precisa de pelo menos um chunk.")
        self.chunks = chunks
        self.embeddings = embeddings
        self.vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        if len(self.vectors) != len(chunks):
            raise RuntimeError("Quantidade de embeddings diferente da quantidade de chunks.")
        dimensions = {len(vector) for vector in self.vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise RuntimeError(
                f"Embeddings com dimensões inconsistentes no índice: {sorted(dimensions)}. "
                "Verifique se o modelo de embedding foi trocado sem reindexar."
            )
        self.dimension = dimensions.pop()

    def search(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        query_vector = self.embeddings.embed_query(query)
        if len(query_vector) != self.dimension:
            raise RuntimeError(
                f"Embedding da consulta tem dimensão {len(query_vector)}; o índice foi construído com {self.dimension}."
            )
        ranked = [
            RetrievedChunk(chunk=chunk, score=float(cosine_similarity(query_vector, vector)))
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(1, k)]
