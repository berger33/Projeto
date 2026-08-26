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
