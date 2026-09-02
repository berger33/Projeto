from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Texto canônico de recusa: único ponto de definição (Fase 2, G-15/R-13). Toda recusa devolvida pela API
# usa exatamente este texto; o que o modelo escreveu fica apenas no log (DEBUG).
REFUSAL_TEXT = "Não encontrei informação suficiente na documentação oficial da Aurora Moda Online."


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED_NO_CONTEXT = "refused_no_context"  # nenhum chunk passou nos filtros de retrieval
    REFUSED_BY_MODEL = "refused_by_model"  # gerador recusou, declarou-se sem sustentação ou falhou na verificação
    ERROR = "error"  # usado pela avaliação quando a execução lança exceção

    @property
    def refused(self) -> bool:
        return self in (AnswerStatus.REFUSED_NO_CONTEXT, AnswerStatus.REFUSED_BY_MODEL)


class Confidence(StrEnum):
    ALTA = "alta"
    MEDIA = "média"
    BAIXA = "baixa"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    locator: dict[str, Any] = field(default_factory=dict)
    # Metadados de chunking (P1-03): seção detectada, posição no texto normalizado da página/arquivo
    # e tamanho estimado em tokens. Opcionais para manter compatibilidade com chunks de CSV/testes.
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_estimate: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class SourceRef:
    document: str
    page: int | None = None
    row: int | None = None
    # Rastreabilidade (P2-01): chunk exato, score vetorial, seção e trecho (<= 200 chars) que sustentou
    # a resposta. ``inferred=True`` quando a fonte veio dos chunks selecionados por falta de citação
    # explícita do gerador (fallback), e não de ``used_sources``/marcadores ``[n]``.
    chunk_id: str | None = None
    score: float | None = None
    section: str | None = None
    excerpt: str | None = None
    inferred: bool = False

    @property
    def identity(self) -> tuple[str, int | None, int | None]:
        return (self.document, self.page, self.row)


@dataclass(frozen=True)
class Generation:
    """Saída de um ``AnswerGenerator``.

    ``refused``/``grounded``/``used_sources`` são sinais **declarados pelo gerador** (saída estruturada);
    ``None``/vazio significa "não informado" e a decisão fica com a verificação independente em
    ``app.refusal``. ``used_sources`` são índices 1-based das ``[FONTE n]`` do contexto.
    """

    text: str
    refused: bool | None = None
    grounded: bool | None = None
    used_sources: tuple[int, ...] = ()
    structured: bool = False
    done_reason: str | None = None


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SourceRef]
    confidence: Confidence
    mode: str
    request_id: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    status: AnswerStatus = AnswerStatus.ANSWERED
    # Diagnóstico da decisão (também vai para o log): por que recusou e a fração da resposta sustentada
    # pelo contexto (``None`` quando não avaliada, ex.: recusa sem contexto ou resposta curta demais).
    refusal_reason: str | None = None
    support: float | None = None


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
