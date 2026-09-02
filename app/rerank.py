"""Diversificação (MMR) e reranking opcional dos candidatos aprovados (Fase 2, R-12; decisão D3).

- ``mmr()`` — *Maximal Marginal Relevance* sobre os candidatos já filtrados: escolhe iterativamente o
  chunk que maximiza ``lambda * relevancia - (1 - lambda) * max_similaridade_com_ja_escolhidos``. Evita que
  quase-duplicatas (``faq.csv:r3`` e ``faq.pdf:p1:c2`` dizem a mesma coisa) ocupem duas vagas do top-k
  quando há outro trecho relevante esperando. A similaridade entre chunks é o cosseno dos vetores do
  índice — sem chamadas extras ao provider.
- ``Reranker`` — interface para um reranker real (cross-encoder local, ``OllamaReranker``…). A
  implementação padrão é ``NoopReranker``; um reranker de verdade só entra por configuração
  (``RAG_RERANKER``) e depois de o eval mostrar ganho >= 5 p.p. em MRR (D3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import RetrievedChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, k: int) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Mantém a ordem recebida (padrão)."""

    name = "noop"

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, k: int) -> list[RetrievedChunk]:
        return candidates[:k]


def mmr(
    candidates: Sequence[RetrievedChunk],
    vectors: Sequence[NDArray[np.float32] | Sequence[float]],
    *,
    k: int,
    lambda_: float = 0.7,
    relevance: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Reordena/corta ``candidates`` por MMR.

    ``vectors[i]`` é o vetor (normalizado ou não) do candidato ``i``; ``relevance[i]`` é o score de
    relevância a usar (padrão: ``candidates[i].score``). ``lambda_=1`` reproduz a ordem por relevância.
    """
    if not candidates or k <= 0:
        return []
    if len(candidates) == 1 or lambda_ >= 1.0:
        return list(candidates[:k])
    matrix = np.asarray([np.asarray(vector, dtype=np.float32) for vector in vectors], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    similarity = matrix @ matrix.T
    scores = np.asarray(relevance if relevance is not None else [item.score for item in candidates], dtype=np.float32)
    # Relevâncias em [0, 1] relativas ao melhor candidato (preserva a proporção entre eles; min-max sobre
    # poucos candidatos zeraria o último e exageraria diferenças pequenas de RRF).
    top = float(scores.max())
    rel = np.clip(scores / top, 0.0, 1.0) if top > 0 else np.ones_like(scores)

    chosen: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(chosen) < k:
        if not chosen:
            best = max(remaining, key=lambda index: (rel[index], -index))
        else:
            penalties = similarity[np.ix_(remaining, chosen)].max(axis=1)
            marginal = lambda_ * rel[remaining] - (1.0 - lambda_) * penalties
            best = remaining[int(np.argmax(marginal))]
        chosen.append(best)
        remaining.remove(best)
    return [candidates[index] for index in chosen]


RERANKERS: dict[str, type[NoopReranker]] = {"noop": NoopReranker}


def build_reranker(name: str) -> Reranker:
    """Instancia o reranker configurado (``RAG_RERANKER``). Nomes desconhecidos falham no boot."""
    try:
        return RERANKERS[name]()
    except KeyError as exc:
        raise ValueError(f"RAG_RERANKER={name!r}: opções disponíveis {sorted(RERANKERS)}") from exc
