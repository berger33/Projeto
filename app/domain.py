from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    locator: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class SourceRef:
    document: str
    page: int | None = None
    row: int | None = None


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SourceRef]
    confidence: str
    mode: str
    request_id: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    # "answered" | "refused_no_context" | "refused_by_model" (vira Enum em P1-01)
    status: str = "answered"


@dataclass(frozen=True)
class Retrieval:
    """Rastro do retrieval de uma pergunta: candidatos devolvidos pelo índice (top-k, em ordem)
    e o subconjunto que passou nos filtros e foi entregue ao gerador."""

    candidates: list[RetrievedChunk]
    selected: list[RetrievedChunk]


@dataclass(frozen=True)
class RAGRun:
    """Resultado completo de uma execução do pipeline: resposta final + rastro do retrieval.
    Usado por avaliação e diagnóstico; a API expõe apenas ``answer``."""

    answer: RAGAnswer
    retrieval: Retrieval
