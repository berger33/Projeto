"""Vector store: interface plugável + implementação ``numpy`` (Fase 2, R-08/R-21; decisão D4).

``VectorStore`` é o ponto de troca de backend (pgvector, Qdrant, …). ``NumpyVectorStore`` guarda os
vetores normalizados numa matriz ``float32`` e responde a ``search`` com um produto matriz-vetor
(``M @ q``) seguido de ``argpartition`` — 10k x 768 em < 1 ms, contra ~80 ms/1k chunks do loop em
Python puro que existia antes. ``filter`` restringe a busca por metadados do chunk (``source``,
``section`` ou qualquer chave do ``locator``) e é o gancho para ACL por documento no futuro.

Persistência (P2-03): ``save``/``load`` gravam a matriz em ``.npy`` e os chunks em JSON; o manifesto
que descreve o índice fica em ``app/persistence.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Chunk

Filter = Mapping[str, object] | Callable[[Chunk], bool] | None


class VectorStore(Protocol):
    """Contrato mínimo de um armazenamento vetorial."""

    @property
    def dimension(self) -> int: ...

    @property
    def chunks(self) -> list[Chunk]: ...

    def __len__(self) -> int: ...

    def scores(self, query_vector: Sequence[float]) -> list[float]: ...

    def search(
        self, query_vector: Sequence[float], *, k: int = 5, filter: Filter = None
    ) -> list[tuple[int, float]]: ...

    def save(self, directory: Path) -> None: ...


def _matches(chunk: Chunk, filter: Filter) -> bool:
    if filter is None:
        return True
    if callable(filter):
        return bool(filter(chunk))
    for key, expected in filter.items():
        actual: object
        if key == "source":
            actual = chunk.source
        elif key == "section":
            actual = chunk.section
        elif key == "id":
            actual = chunk.id
        else:
            actual = chunk.locator.get(key)
        if isinstance(expected, list | tuple | set | frozenset):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class NumpyVectorStore:
    """Matriz ``(n, d)`` float32 com linhas normalizadas; cosseno = produto interno."""

    VECTORS_FILE = "vectors.npy"
    CHUNKS_FILE = "chunks.json"

    def __init__(self, chunks: list[Chunk], vectors: Sequence[Sequence[float]] | NDArray[np.floating]) -> None:
        if not chunks:
            raise ValueError("O índice precisa de pelo menos um chunk.")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise RuntimeError(
                f"Quantidade de embeddings ({matrix.shape[0] if matrix.ndim == 2 else 'n/d'}) diferente da "
                f"quantidade de chunks ({len(chunks)})."
            )
        if matrix.shape[1] == 0:
            raise RuntimeError("Embeddings com dimensão zero.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix: NDArray[np.float32] = (matrix / norms).astype(np.float32)
        self._chunks = list(chunks)

    # ----- VectorStore -----

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def matrix(self) -> NDArray[np.float32]:
        return self._matrix

    def __len__(self) -> int:
        return len(self._chunks)

    def _query(self, query_vector: Sequence[float]) -> NDArray[np.float32]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise RuntimeError(
                f"Embedding da consulta tem dimensão {query.shape[0] if query.ndim == 1 else 'n/d'}; "
                f"o índice foi construído com {self.dimension}."
            )
        norm = float(np.linalg.norm(query))
        return query / norm if norm else query

    def scores(self, query_vector: Sequence[float]) -> list[float]:
        """Cosseno com todos os chunks, na ordem do índice."""
        return [float(value) for value in self._matrix @ self._query(query_vector)]

    def search(self, query_vector: Sequence[float], *, k: int = 5, filter: Filter = None) -> list[tuple[int, float]]:
        """``[(índice do chunk, cosseno)]`` dos ``k`` mais próximos, em ordem decrescente."""
        similarities = self._matrix @ self._query(query_vector)
        if filter is not None:
            mask = np.fromiter((_matches(chunk, filter) for chunk in self._chunks), dtype=bool, count=len(self._chunks))
            if not mask.any():
                return []
            similarities = np.where(mask, similarities, -np.inf)
        k = max(1, min(k, len(self._chunks)))
        top = np.argpartition(-similarities, k - 1)[:k] if k < len(self._chunks) else np.arange(len(self._chunks))
        ordered = top[np.argsort(-similarities[top], kind="stable")]
        return [(int(index), float(similarities[index])) for index in ordered if np.isfinite(similarities[index])]

    # ----- persistência -----

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self.VECTORS_FILE, self._matrix)
        payload = [asdict(chunk) for chunk in self._chunks]
        (directory / self.CHUNKS_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        matrix = np.load(directory / cls.VECTORS_FILE)
        raw = json.loads((directory / cls.CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(chunks, matrix)
