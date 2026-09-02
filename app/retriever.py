"""Retrieval híbrido: vetorial + BM25 fundidos por RRF, na ordem correta (Fase 2, R-09/R-10).

Pipeline por consulta:

1. Recupera ``candidate_pool = pool_factor * k`` candidatos de **cada** canal (cosseno no ``VectorIndex``
   e BM25 no ``BM25Index``).
2. Funde por *Reciprocal Rank Fusion*: ``score = Σ 1 / (rrf_k + rank_canal)``. RRF é robusto às
   escalas incomparáveis dos dois canais (cosseno em [-1, 1] vs. BM25 sem limite superior).
3. Aplica os filtros sobre o **conjunto fundido** (não sobre um top-k já cortado):
   - ``min_score`` no cosseno (guarda-chuva contra vetores aleatórios);
   - sinal lexical: um candidato só sobrevive se compartilha ao menos um radical de conteúdo com a
     pergunta **ou** se o canal vetorial o colocou no topo com folga (``vector_only_min_score``) —
     o gate deixa de ser veto absoluto e vira uma das condições.
4. Corta em ``k``.

Cada ``RetrievedChunk.score`` continua sendo o **cosseno** (compatível com os limiares de confiança
existentes); a fusão é usada para ordenar e está disponível em ``ScoredCandidate`` para diagnóstico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domain import Chunk, RetrievedChunk
from .lexical import BM25Index
from .rerank import NoopReranker, Reranker, mmr
from .retrieval import VectorIndex
from .store import NumpyVectorStore


@dataclass(frozen=True)
class ScoredCandidate:
    chunk: Chunk
    vector_score: float  # cosseno com a consulta (calculado para todo candidato)
    vector_rank: int | None  # posição no canal vetorial, se dentro do pool
    lexical_score: float  # BM25 (0 se não veio do canal lexical)
    lexical_rank: int | None
    lexical_overlap: int  # radicais da pergunta presentes no chunk
    lexical_coverage: float  # fração (ponderada por IDF) dos radicais da pergunta presentes no chunk
    fused_score: float  # RRF


@dataclass(frozen=True)
class RetrieverConfig:
    k: int = 5
    pool_factor: int = 4
    rrf_k: int = 60
    min_score: float = 0.12
    # Cosseno a partir do qual um candidato é aceito mesmo sem sobreposição lexical (sinônimos/paráfrases
    # que embeddings semânticos capturam e BM25 não). Com o hash local esse caminho raramente dispara.
    vector_only_min_score: float = 0.5
    # Cobertura lexical mínima (ponderada por IDF) para aceitar um candidato com cosseno >= min_score.
    # Um único termo genérico em comum ("loja", "conta") não basta; termos raros da pergunta que o
    # chunk não tem pesam contra. Calibrado pelo eval em modo local (P1-06 refina por provider).
    min_lexical_coverage: float = 0.2
    # Cosseno a partir do qual basta **um** radical em comum (paráfrases com um termo-âncora específico,
    # ex.: "reembolso" em "quanto tempo demora para o reembolso aparecer na fatura").
    vector_with_overlap_min_score: float = 0.35
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    # Diversificação MMR sobre os aprovados (1.0 = desligada; 0.7 = relevância pesa 70 %). Default por perfil.
    mmr_lambda: float = 1.0


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vector_index: VectorIndex,
        config: RetrieverConfig | None = None,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks = chunks
        self.vector_index = vector_index
        self.lexical_index = BM25Index([chunk.text for chunk in chunks])
        self.config = config or RetrieverConfig()
        self.reranker: Reranker = reranker or NoopReranker()
        self._position = {chunk.id: index for index, chunk in enumerate(chunks)}

    # ----- canais -----

    def fuse(self, query: str) -> list[ScoredCandidate]:
        """Candidatos dos dois canais fundidos por RRF, em ordem decrescente de ``fused_score``."""
        config = self.config
        pool = max(config.k * config.pool_factor, config.k)
        vector_scores = self.vector_index.scores(query)
        vector_order = sorted(range(len(self.chunks)), key=lambda index: vector_scores[index], reverse=True)
        vector_rank = {index: rank for rank, index in enumerate(vector_order[:pool], start=1)}
        lexical_hits = self.lexical_index.search(query, k=pool)
        lexical_rank = {hit.index: rank for rank, hit in enumerate(lexical_hits, start=1)}
        lexical_score = {hit.index: hit.score for hit in lexical_hits}

        candidates: list[ScoredCandidate] = []
        for index in set(vector_rank) | set(lexical_rank):
            fused = 0.0
            if index in vector_rank:
                fused += config.vector_weight / (config.rrf_k + vector_rank[index])
            if index in lexical_rank:
                fused += config.lexical_weight / (config.rrf_k + lexical_rank[index])
            candidates.append(
                ScoredCandidate(
                    chunk=self.chunks[index],
                    vector_score=vector_scores[index],
                    vector_rank=vector_rank.get(index),
                    lexical_score=lexical_score.get(index, 0.0),
                    lexical_rank=lexical_rank.get(index),
                    lexical_overlap=self.lexical_index.overlap(query, index),
                    lexical_coverage=self.lexical_index.coverage(query, index),
                    fused_score=fused,
                )
            )
        candidates.sort(key=lambda item: (item.fused_score, item.vector_score), reverse=True)
        return candidates

    # ----- filtros -----

    def accepts(self, candidate: ScoredCandidate) -> bool:
        """Três níveis de evidência, do mais lexical ao mais vetorial:

        1. cobertura lexical >= ``min_lexical_coverage`` e cosseno >= ``min_score``;
        2. ao menos um radical em comum e cosseno >= ``vector_with_overlap_min_score``;
        3. cosseno >= ``vector_only_min_score`` (sem exigência lexical).
        """
        config = self.config
        if candidate.vector_score < config.min_score:
            return False
        if candidate.lexical_coverage >= config.min_lexical_coverage:
            return True
        if candidate.lexical_overlap >= 1 and candidate.vector_score >= config.vector_with_overlap_min_score:
            return True
        return candidate.vector_score >= config.vector_only_min_score

    def select(self, query: str, fused: list[ScoredCandidate]) -> list[RetrievedChunk]:
        """Filtra o pool fundido, diversifica por MMR e aplica o reranker; devolve no máximo ``k``."""
        approved = [RetrievedChunk(chunk=item.chunk, score=item.vector_score) for item in fused if self.accepts(item)]
        if not approved:
            return []
        approved_fused = [item.fused_score for item in fused if self.accepts(item)]
        # Relevância para o MMR = score de fusão (RRF), que já combina os dois canais; o cosseno segue
        # como ``score`` do RetrievedChunk para limiares e confiança.
        if self.config.mmr_lambda < 1.0 and len(approved) > 1:
            vectors = self._vectors_for(approved)
            approved = mmr(approved, vectors, k=len(approved), lambda_=self.config.mmr_lambda, relevance=approved_fused)
        return self.reranker.rerank(query, approved, k=self.config.k)

    def _vectors_for(self, items: list[RetrievedChunk]) -> list[NDArray[np.float32]]:
        store = self.vector_index.store
        if isinstance(store, NumpyVectorStore):
            return [store.matrix[self._position[item.chunk.id]] for item in items]
        # Backend sem acesso direto à matriz: recorre ao provider (custo de embed por chunk).
        return [
            np.asarray(vector, dtype=np.float32)
            for vector in self.vector_index.embeddings.embed_documents([item.chunk.text for item in items])
        ]

    def retrieve(self, query: str) -> tuple[list[ScoredCandidate], list[RetrievedChunk]]:
        """``(candidatos fundidos, selecionados)`` — selecionados filtrados, diversificados e cortados em ``k``."""
        candidates = self.fuse(query)
        return candidates, self.select(query, candidates)
