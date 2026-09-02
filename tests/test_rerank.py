"""P2-05: MMR (diversidade) + interface Reranker com implementação Noop (R-12; decisão D3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import ConfigError, Settings
from app.domain import Chunk, RetrievedChunk
from app.embeddings import HashEmbeddingProvider
from app.rag import RAGService
from app.rerank import RERANKERS, NoopReranker, Reranker, build_reranker, mmr
from app.retrieval import VectorIndex
from app.retriever import HybridRetriever, RetrieverConfig

ROOT = Path(__file__).resolve().parents[1]


def _item(chunk_id: str, score: float, text: str = "t") -> RetrievedChunk:
    return RetrievedChunk(Chunk(id=chunk_id, text=text, source=chunk_id.split(":")[0], locator={}), score)


# ---------------------------------------------------------------------------
# mmr()
# ---------------------------------------------------------------------------


def test_mmr_penalizes_near_duplicates_and_promotes_diverse_relevant_chunk() -> None:
    # a e b são quase idênticos (vetores paralelos); c é diferente e um pouco menos relevante.
    items = [_item("a:1", 0.9), _item("b:1", 0.88), _item("c:1", 0.8), _item("d:1", 0.2)]
    vectors = [np.array([1.0, 0.0]), np.array([0.99, 0.1]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    by_relevance = mmr(items, vectors, k=2, lambda_=1.0)
    assert [i.chunk.id for i in by_relevance] == ["a:1", "b:1"]
    diversified = mmr(items, vectors, k=2, lambda_=0.7)
    assert [i.chunk.id for i in diversified] == ["a:1", "c:1"]
    # k maior que o nº de candidatos devolve todos, ainda reordenados.
    assert {i.chunk.id for i in mmr(items, vectors, k=10, lambda_=0.7)} == {"a:1", "b:1", "c:1", "d:1"}


def test_mmr_edge_cases() -> None:
    assert mmr([], [], k=3) == []
    single = [_item("a:1", 0.5)]
    assert mmr(single, [np.array([1.0])], k=3) == single
    assert mmr(single, [np.array([1.0])], k=0) == []
    same = [_item("a:1", 0.5), _item("b:1", 0.5)]  # relevâncias iguais → span 0 → todas 1.0; ordem estável
    out = mmr(same, [np.array([1.0, 0.0]), np.array([0.0, 1.0])], k=2, lambda_=0.7)
    assert [i.chunk.id for i in out] == ["a:1", "b:1"]
    zero = mmr(same, [np.array([0.0, 0.0]), np.array([0.0, 0.0])], k=2, lambda_=0.5)  # vetores nulos não geram NaN
    assert len(zero) == 2


def test_mmr_uses_explicit_relevance_when_given() -> None:
    items = [_item("a:1", 0.1), _item("b:1", 0.9)]
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert mmr(items, vectors, k=1, lambda_=0.7, relevance=[0.9, 0.1])[0].chunk.id == "a:1"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def test_noop_reranker_keeps_order_and_cuts_k() -> None:
    items = [_item("a:1", 0.3), _item("b:1", 0.9)]
    assert NoopReranker().rerank("q", items, k=1) == items[:1]
    reranker: Reranker = build_reranker("noop")
    assert reranker.name == "noop" and "noop" in RERANKERS


def test_unknown_reranker_fails_at_boot() -> None:
    with pytest.raises(ValueError, match="RAG_RERANKER='cross-encoder'"):
        build_reranker("cross-encoder")
    with pytest.raises(ConfigError, match="RAG_MMR_LAMBDA"):
        Settings(mmr_lambda=1.5)
    with pytest.raises(ConfigError, match="RAG_RERANKER"):
        Settings(reranker="  ")


def test_custom_reranker_is_pluggable_in_retriever() -> None:
    class Reverse:
        name = "reverse"

        def rerank(self, query: str, candidates: list[RetrievedChunk], *, k: int) -> list[RetrievedChunk]:
            return list(reversed(candidates))[:k]

    corpus = [
        Chunk(id="a.csv:r2", text="prazo de devolução 10 dias corridos", source="a.csv", locator={"row": 2}),
        Chunk(id="b.csv:r2", text="devolução do produto sem sinais de uso", source="b.csv", locator={"row": 2}),
        Chunk(id="c.csv:r2", text="pagamento cartão pix", source="c.csv", locator={"row": 2}),
    ]
    retriever = HybridRetriever(
        corpus,
        VectorIndex(corpus, HashEmbeddingProvider()),
        RetrieverConfig(k=2, min_score=0.0, mmr_lambda=1.0),
        reranker=Reverse(),
    )
    fused = retriever.fuse("devolução")
    approved = [item.chunk.id for item in fused if retriever.accepts(item)]
    selected = [item.chunk.id for item in retriever.select("devolução", fused)]
    assert selected == list(reversed(approved))[:2]


# ---------------------------------------------------------------------------
# Integração no serviço
# ---------------------------------------------------------------------------


def test_mmr_is_applied_after_filters_and_within_k() -> None:
    """Com dois quase-duplicados no topo do pool fundido, a 2ª vaga vai para o trecho diverso."""
    from app.retriever import ScoredCandidate

    corpus = [
        Chunk(
            id="a.csv:r2",
            text="O prazo de devolução é de 10 dias corridos após o recebimento do pedido.",
            source="a.csv",
            locator={"row": 2},
        ),
        Chunk(
            id="a.csv:r3",
            text="O prazo de devolução é de 10 dias corridos após o recebimento do pedido pelo cliente.",
            source="a.csv",
            locator={"row": 3},
        ),
        Chunk(
            id="a.csv:r4",
            text="O produto deve estar sem sinais de uso e com etiquetas para devolução.",
            source="a.csv",
            locator={"row": 4},
        ),
        Chunk(id="a.csv:r5", text="Aceitamos PIX e cartão.", source="a.csv", locator={"row": 5}),
    ]
    index = VectorIndex(corpus, HashEmbeddingProvider())

    def candidate(chunk: Chunk, cos: float, rrf: float) -> ScoredCandidate:
        return ScoredCandidate(chunk, cos, 1, 1.0, 1, 2, 0.6, rrf)

    fused = [candidate(corpus[0], 0.46, 0.033), candidate(corpus[1], 0.45, 0.032), candidate(corpus[2], 0.30, 0.031)]
    plain = HybridRetriever(corpus, index, RetrieverConfig(k=2, min_score=0.0, mmr_lambda=1.0))
    diversified = HybridRetriever(corpus, index, RetrieverConfig(k=2, min_score=0.0, mmr_lambda=0.7))
    assert [item.chunk.id for item in plain.select("q", fused)] == ["a.csv:r2", "a.csv:r3"]
    assert [item.chunk.id for item in diversified.select("q", fused)] == ["a.csv:r2", "a.csv:r4"]
    # O score exposto continua sendo o cosseno do candidato (não o RRF).
    assert diversified.select("q", fused)[0].score == 0.46


def test_mmr_lambda_comes_from_profile_with_env_override() -> None:
    from app.config import THRESHOLD_PROFILES

    assert (
        Settings().mmr_lambda is None
        and Settings().thresholds.mmr_lambda == THRESHOLD_PROFILES["local"].mmr_lambda == 1.0
    )
    assert Settings(rag_mode="ollama").thresholds.mmr_lambda == 0.7
    custom = Settings.from_env({"RAG_MMR_LAMBDA": "0.6", "RAG_RERANKER": "NOOP"})
    assert custom.mmr_lambda == 0.6 and custom.thresholds.mmr_lambda == 0.6 and custom.reranker == "noop"
    service = RAGService(ROOT / "docs", custom)
    assert service.retriever.config.mmr_lambda == 0.6 and service.retriever.reranker.name == "noop"
    assert custom.public_dict()["thresholds"]["mmr_lambda"] == 0.6 and custom.public_dict()["reranker"] == "noop"


def test_mmr_does_not_change_refusals_or_eval_gates() -> None:
    service = RAGService(ROOT / "docs", Settings())
    assert service.answer("Qual é a capital da Austrália?").status == "refused_no_context"
    result = service.run("Qual é o prazo para devolver uma compra?")
    assert result.answer.status == "answered" and 1 <= len(result.retrieval.selected) <= 5
