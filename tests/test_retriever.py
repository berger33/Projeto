"""P1-04: ordem do pipeline + busca híbrida (BM25 + vetorial, fusão RRF) + normalização PT-BR.

Findings: R-09 (top-k cortado antes do filtro), R-10 (sem híbrido, sem normalização de acentos,
18 stopwords, gate lexical como veto), G-15 (tokenizador único).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ConfigError, Settings
from app.domain import Chunk
from app.embeddings import HashEmbeddingProvider
from app.lexical import BM25Index, analyze, stem
from app.rag import RAGService
from app.retrieval import VectorIndex
from app.retriever import HybridRetriever, RetrieverConfig, ScoredCandidate

ROOT = Path(__file__).resolve().parents[1]


def _chunk(chunk_id: str, text: str) -> Chunk:
    source, _, _ = chunk_id.partition(":")
    return Chunk(id=chunk_id, text=text, source=source, locator={"row": 1})


CORPUS = [
    _chunk("faq.csv:r2", "Devolução: o prazo para devolver é de 10 dias corridos após o recebimento do produto."),
    _chunk(
        "faq.csv:r3",
        "Pagamento: aceitamos cartão de crédito e PIX; o pagamento por PIX é confirmado após identificação.",
    ),
    _chunk("faq.csv:r4", "Entrega: o prazo de entrega começa após a confirmação do pagamento e varia conforme o CEP."),
    _chunk("faq.csv:r5", "Rastreamento: após o despacho o cliente recebe informações de acompanhamento do pedido."),
    _chunk("faq.csv:r6", "Suporte: contato pelo e-mail suporte@auroramoda.exemplo com o número do pedido."),
    _chunk(
        "politica.pdf:p1:c1",
        "Reembolso: após a aprovação da devolução, o reembolso é feito no mesmo meio de pagamento.",
    ),
    _chunk(
        "politica.pdf:p1:c2",
        "Privacidade: dados pessoais como nome, CPF e endereço são coletados para processar pedidos.",
    ),
]


@pytest.fixture()
def retriever() -> HybridRetriever:
    return HybridRetriever(CORPUS, VectorIndex(CORPUS, HashEmbeddingProvider()), RetrieverConfig(k=3, min_score=0.05))


# ---------------------------------------------------------------------------
# Normalização e stemmer (R-10, G-15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("devolução", "devolucao"),
        ("devolução", "devoluções"),
        ("devolver", "devolvido"),
        ("reembolso", "reembolsos"),
        ("pagamento", "pagar"),
        ("entrega", "entregas"),
        ("rastreamento", "rastrear"),
        ("cartão", "cartoes"),
        ("informação", "informações"),
        ("solicitar", "solicitação"),
    ],
)
def test_stemmer_conflates_inflections(left: str, right: str) -> None:
    assert analyze(left) == analyze(right)


def test_stemmer_keeps_short_tokens_and_numbers() -> None:
    assert stem("pix") == "pix" and stem("cep") == "cep" and stem("10") == "10"
    assert analyze("Qual é o prazo de 10 dias para a devolução?") == ["praz", "10", "dia", "devol"]


def test_bm25_ranks_by_term_relevance_and_ignores_stopwords() -> None:
    index = BM25Index([chunk.text for chunk in CORPUS])
    hits = index.search("qual o prazo de devolucao", k=3)
    assert CORPUS[hits[0].index].id == "faq.csv:r2"
    assert index.search("o a de para com", k=3) == []  # só stopwords
    assert index.search("", k=3) == []
    assert index.overlap("reembolsos", CORPUS.index(CORPUS[5])) == 1
    assert 0.0 < index.coverage("prazo devolucao xyzzy", 0) < 1.0  # termo ausente do corpus penaliza
    assert index.coverage("prazo devolucao", 0) == 1.0


def test_bm25_handles_empty_corpus() -> None:
    index = BM25Index([])
    assert index.search("qualquer", k=3) == [] and len(index) == 0


# ---------------------------------------------------------------------------
# Fusão RRF e ordem do pipeline (R-09)
# ---------------------------------------------------------------------------


def test_fuse_combines_both_channels_with_rrf(retriever: HybridRetriever) -> None:
    candidates = retriever.fuse("qual o prazo de devolucao")
    assert candidates[0].chunk.id == "faq.csv:r2"
    top = candidates[0]
    assert top.vector_rank is not None and top.lexical_rank is not None
    expected = 1 / (60 + top.vector_rank) + 1 / (60 + top.lexical_rank)
    assert top.fused_score == pytest.approx(expected)
    assert top.lexical_overlap >= 2 and top.lexical_coverage > 0.5
    # Pool maior que k: candidatos além da posição k continuam disponíveis para o filtro.
    assert len(candidates) > retriever.config.k


def test_filter_runs_over_the_fused_pool_not_over_a_truncated_top_k() -> None:
    """R-09: um candidato válido em posição > k no canal vetorial não pode ser descartado antes do filtro."""
    config = RetrieverConfig(k=1, pool_factor=4, min_score=0.0, min_lexical_coverage=0.2)
    retriever = HybridRetriever(CORPUS, VectorIndex(CORPUS, HashEmbeddingProvider()), config)
    candidates, selected = retriever.retrieve("reembolso no mesmo meio de pagamento")
    assert len(candidates) >= 4  # pool = 4 * k
    assert len(selected) == 1 and selected[0].chunk.id == "politica.pdf:p1:c1"


def test_accepts_requires_evidence_levels() -> None:
    config = RetrieverConfig(
        min_score=0.12, min_lexical_coverage=0.2, vector_with_overlap_min_score=0.35, vector_only_min_score=0.5
    )
    retriever = HybridRetriever(CORPUS, VectorIndex(CORPUS, HashEmbeddingProvider()), config)

    def candidate(cos: float, overlap: int, coverage: float) -> ScoredCandidate:
        return ScoredCandidate(CORPUS[0], cos, 1, 1.0, 1, overlap, coverage, 0.03)

    assert not retriever.accepts(candidate(0.10, 3, 1.0))  # abaixo do piso de cosseno
    assert retriever.accepts(candidate(0.15, 1, 0.25))  # cobertura suficiente
    assert not retriever.accepts(candidate(0.15, 1, 0.10))  # um termo genérico só
    assert retriever.accepts(candidate(0.40, 1, 0.10))  # cosseno médio + um radical âncora
    assert not retriever.accepts(candidate(0.40, 0, 0.0))  # cosseno médio sem nada lexical
    assert retriever.accepts(candidate(0.55, 0, 0.0))  # cosseno alto sozinho


def test_accents_and_inflections_no_longer_block_retrieval() -> None:
    service = RAGService(ROOT / "corpus", Settings())
    for question in ("qual o prazo de devolucao", "posso pagar com cartao", "reembolsos demoram quanto"):
        result = service.run(question)
        assert result.retrieval.selected, question
        assert result.answer.status == "answered", question


def test_out_of_scope_questions_sharing_generic_terms_are_refused() -> None:
    service = RAGService(ROOT / "corpus", Settings())
    for question in (
        "Qual é o horário de funcionamento da loja física no shopping?",
        "Como faço para trocar a senha da minha conta?",
        "Vocês fazem entrega internacional para Portugal?",
        "Quanto custa a camiseta básica branca tamanho M?",
        "Qual é a capital da Austrália?",
    ):
        result = service.answer(question)
        assert result.status == "refused_no_context" and result.sources == [], question


def test_retrieved_event_exposes_fusion_diagnostics(captured) -> None:
    service = RAGService(ROOT / "corpus", Settings())
    service.answer("Qual é o prazo para devolver uma compra?")
    (event,) = captured.events("query.retrieved")
    assert event["pool"] >= 5
    first = event["candidates"][0]
    assert {"id", "score", "vector_rank", "lexical_rank", "overlap", "coverage", "rrf"} <= set(first)
    assert event["selected"][0] in {"faq.csv:r7", "politica_reembolso_devolucoes.pdf:p1:c1", "faq.pdf:p1:c6"}


def test_vector_only_threshold_is_configurable_and_validated() -> None:
    settings = Settings.from_env({"RAG_VECTOR_ONLY_MIN_SCORE": "0.7", "RAG_INDEX_DIR": ""})
    assert settings.vector_only_min_score == 0.7
    with pytest.raises(ConfigError, match="RAG_VECTOR_ONLY_MIN_SCORE"):
        Settings(vector_only_min_score=1.5)
    service = RAGService(ROOT / "corpus", settings)
    assert service.retriever.config.vector_only_min_score == 0.7
    assert service.retriever.config.k == settings.retrieval_k
