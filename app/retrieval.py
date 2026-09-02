"""Índice vetorial: embeddings do corpus + ``VectorStore`` (numpy por padrão)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .domain import Chunk, RetrievedChunk
from .embeddings import EmbeddingProvider
from .store import Filter, NumpyVectorStore, VectorStore


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosseno puro-Python (referência/testes; a busca usa ``VectorStore``)."""
    if len(left) != len(right):
        raise ValueError("Vetores com dimensões diferentes.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorIndex:
    """Liga um ``EmbeddingProvider`` a um ``VectorStore``.

    ``VectorIndex(chunks, embeddings)`` embeda os chunks e constrói um ``NumpyVectorStore``;
    ``VectorIndex.from_store(store, embeddings)`` reutiliza um store já carregado do disco (P2-03).
    """

    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider, *, store: VectorStore | None = None):
        if not chunks:
            raise ValueError("O índice precisa de pelo menos um chunk.")
        self.embeddings = embeddings
        if store is None:
            vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("Quantidade de embeddings diferente da quantidade de chunks.")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1 or 0 in dimensions:
                raise RuntimeError(
                    f"Embeddings com dimensões inconsistentes no índice: {sorted(dimensions)}. "
                    "Verifique se o modelo de embedding foi trocado sem reindexar."
                )
            store = NumpyVectorStore(chunks, vectors)
        self.store: VectorStore = store

    @classmethod
    def from_store(cls, store: VectorStore, embeddings: EmbeddingProvider) -> VectorIndex:
        return cls(store.chunks, embeddings, store=store)

    @property
    def chunks(self) -> list[Chunk]:
        return self.store.chunks

    @property
    def dimension(self) -> int:
        return self.store.dimension

    def scores(self, query: str) -> list[float]:
        """Cosseno entre a consulta e **todos** os chunks, na ordem do índice."""
        if not query.strip():
            return [0.0] * len(self.chunks)
        return self.store.scores(self.embeddings.embed_query(query))

    def search(self, query: str, *, k: int = 5, filter: Filter = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        hits = self.store.search(self.embeddings.embed_query(query), k=k, filter=filter)
        return [RetrievedChunk(chunk=self.chunks[index], score=score) for index, score in hits]
