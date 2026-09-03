"""P2-01: fontes derivadas do uso real + citações inline + SourceResponse com chunk_id/score/excerpt.

Findings: R-17 (fontes = 3 primeiros selecionados, política de privacidade citada para pagamento),
R-16 (marcadores [FONTE n] nunca pedidos de volta), G-28 (SourceRef sem chunk_id/score/trecho).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.config import Settings
from app.domain import Chunk, Generation, RetrievedChunk
from app.generation import SYSTEM_PROMPT
from app.main import create_app
from app.rag import RAGService
from app.sources import best_excerpt, cited_indexes, derive_sources, strip_citations
from evals.harness import load_cases, run_eval

ROOT = Path(__file__).resolve().parents[1]


def _item(chunk_id: str, text: str, score: float, section: str | None = None) -> RetrievedChunk:
    source, _, rest = chunk_id.partition(":")
    locator = {"row": int(rest[1:])} if rest.startswith("r") else {"page": 1}
    return RetrievedChunk(Chunk(id=chunk_id, text=text, source=source, locator=locator, section=section), score)


SELECTED = [
    _item("faq.csv:r3", "Pagamento: aceitamos cartão de crédito e PIX. O PIX é confirmado após identificação.", 0.44),
    _item(
        "faq.pdf:p1:c2",
        "2. Quais métodos de pagamento são aceitos?\nSão aceitos cartão de crédito e PIX.",
        0.37,
        "FAQ / 2.",
    ),
    _item(
        "politica_privacidade.pdf:p1:c2",
        "Dados completos de cartão não são armazenados pela loja.",
        0.30,
        "Privacidade / 2.",
    ),
]


# ---------------------------------------------------------------------------
# Parsing de citações e trechos
# ---------------------------------------------------------------------------


def test_cited_indexes_parses_single_multiple_and_fonte_markers() -> None:
    assert cited_indexes("Aceitamos PIX [1]. Cartão também [2, 3]; ver [FONTE 1].") == [1, 2, 3]
    assert cited_indexes("Sem citações.") == []
    assert cited_indexes("Lista: [a] [1;2]") == [1, 2]


def test_strip_citations_cleans_text_without_breaking_punctuation() -> None:
    assert strip_citations("O prazo é de 10 dias [1]. Aceitamos PIX [2, 3] e cartão [FONTE 1].") == (
        "O prazo é de 10 dias. Aceitamos PIX e cartão."
    )
    assert strip_citations("Sem marcadores.") == "Sem marcadores."


def test_best_excerpt_picks_most_overlapping_sentence_and_clips() -> None:
    chunk = "Título irrelevante. O prazo para devolução é de 10 dias corridos após o recebimento. Outra frase."
    assert (
        best_excerpt(chunk, "O prazo de devolução é de 10 dias")
        == "O prazo para devolução é de 10 dias corridos após o recebimento."
    )
    long = "palavra " * 100
    excerpt = best_excerpt(long, "nada em comum")
    assert len(excerpt) <= 200 and excerpt.endswith("…")


# ---------------------------------------------------------------------------
# derive_sources: estruturado → inline → fallback
# ---------------------------------------------------------------------------


def test_sources_from_structured_used_sources_only() -> None:
    generation = Generation(text="Aceitamos cartão e PIX.", used_sources=(2,), structured=True)
    sources, reason = derive_sources(generation, SELECTED)
    assert reason == "structured"
    assert [source.chunk_id for source in sources] == ["faq.pdf:p1:c2"]
    source = sources[0]
    assert source.document == "faq.pdf" and source.page == 1 and source.row is None
    assert source.score == 0.37 and source.section == "FAQ / 2." and source.inferred is False
    assert source.excerpt == "São aceitos cartão de crédito e PIX."


def test_sources_from_inline_citations_when_no_structured_list() -> None:
    generation = Generation(text="Aceitamos cartão e PIX [1]. Confirmação após identificação [1].")
    sources, reason = derive_sources(generation, SELECTED)
    assert reason == "inline" and [source.chunk_id for source in sources] == ["faq.csv:r3"]
    assert sources[0].row == 3 and sources[0].inferred is False


def test_sources_fallback_marks_inferred_and_keeps_all_selected() -> None:
    generation = Generation(text="Aceitamos cartão e PIX.")
    sources, reason = derive_sources(generation, SELECTED)
    assert reason == "fallback"
    assert [source.chunk_id for source in sources] == [item.chunk.id for item in SELECTED]
    assert all(source.inferred for source in sources)


def test_sources_ignore_out_of_range_indexes_and_dedupe() -> None:
    generation = Generation(text="x [9] [1] [1]", used_sources=(7, 0))
    sources, reason = derive_sources(generation, SELECTED)
    assert reason == "inline" and [source.chunk_id for source in sources] == ["faq.csv:r3"]
    assert derive_sources(Generation(text="x", used_sources=(1,)), []) == ([], "fallback")


# ---------------------------------------------------------------------------
# Integração: RAGService, API, eval
# ---------------------------------------------------------------------------


@pytest.fixture()
def service() -> RAGService:
    return RAGService(ROOT / "corpus", Settings())


def test_payment_question_no_longer_cites_privacy_policy(service: RAGService) -> None:
    """R-17: a política de privacidade passava no gate por compartilhar 'pagamento' e era citada."""
    result = service.answer("Quais formas de pagamento são aceitas?")
    assert result.status == "answered"
    assert {source.document for source in result.sources} <= {"faq.csv", "faq.pdf"}
    assert all(source.chunk_id and source.score is not None and source.excerpt for source in result.sources)
    assert all(not source.inferred for source in result.sources)  # extrativo declara used_sources


def test_answer_text_has_no_inline_markers_and_sources_are_logged(service: RAGService, captured: ListHandler) -> None:
    class CitingGenerator:
        mode = "stub"

        def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
            return Generation(
                text="O prazo para devolução é de 10 dias corridos após o recebimento [2].", structured=False
            )

    service.generator = CitingGenerator()  # type: ignore[assignment]
    result = service.answer("Qual é o prazo para devolver uma compra?")
    assert "[2]" not in result.answer and result.answer.endswith("recebimento.")
    assert len(result.sources) == 1 and not result.sources[0].inferred
    (event,) = captured.events("query.answered")
    assert event["sources_reason"] == "inline" and event["source_ids"] == [result.sources[0].chunk_id]


def test_api_exposes_source_traceability_fields_additively(service: RAGService) -> None:
    with TestClient(create_app(service)) as client:
        body = client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).json()
    source = body["sources"][0]
    assert {"document", "page", "row"} <= set(source)  # contrato antigo
    assert {"chunk_id", "score", "section", "excerpt", "inferred"} <= set(source)
    assert source["chunk_id"].startswith(source["document"]) and 0 < source["score"] <= 1
    assert isinstance(source["excerpt"], str) and 0 < len(source["excerpt"]) <= 200


def test_prompt_asks_for_inline_citations() -> None:
    assert "[2]" in SYSTEM_PROMPT and "número da fonte entre colchetes" in SYSTEM_PROMPT


def test_source_precision_meets_plan_target_in_local_mode(service: RAGService) -> None:
    metrics = run_eval(service, load_cases()).metrics
    assert metrics["source_precision"] >= 0.9
