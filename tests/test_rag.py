from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.documents import load_chunks, split_text
from app.main import create_app
from app.rag import RAGService


@pytest.fixture()
def docs(tmp_path: Path) -> Path:
    (tmp_path / "faq.csv").write_text(
        "pergunta,resposta\n"
        "formas de pagamento,Aceitamos cartão de crédito e PIX.\n"
        "prazo de entrega,O prazo começa após a confirmação do pagamento.\n"
        "devolução,A devolução pode ser solicitada em até 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def service(docs: Path) -> RAGService:
    return RAGService(docs, Settings(rag_mode="local", min_score=0.05, retrieval_k=3))


def test_split_text_has_overlap_and_no_empty_chunks():
    text = ("Primeiro parágrafo com conteúdo relevante. " * 20) + "\n\n" + ("Segundo parágrafo. " * 30)
    chunks = split_text(text, chunk_size=180, overlap=30)
    assert len(chunks) > 2
    assert all(chunk.strip() for chunk in chunks)


def test_loads_csv_as_traceable_chunks(docs: Path):
    chunks = load_chunks(docs)
    assert len(chunks) == 3
    assert chunks[0].source == "faq.csv"
    assert "row" in chunks[0].locator


def test_retrieval_returns_relevant_source(service: RAGService):
    result = service.answer("Quais formas de pagamento são aceitas?")
    assert "PIX" in result.answer
    assert result.sources
    assert result.sources[0].document == "faq.csv"


def test_out_of_scope_refuses_without_sources(service: RAGService):
    result = service.answer("Qual é a capital da Austrália?")
    assert "não encontrei" in result.answer.lower()
    assert result.sources == []
    assert result.confidence == "baixa"


def test_empty_question_is_rejected_by_domain(service: RAGService):
    with pytest.raises(ValueError):
        service.answer("   ")


def test_api_contract_and_health(service: RAGService):
    client = TestClient(create_app(service))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["chunks"] == 3
    assert health.json()["mode"] == "local-extractive"

    response = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert response.status_code == 200
    payload = response.json()
    assert "10 dias" in payload["answer"]
    assert payload["sources"][0]["document"] == "faq.csv"


def test_api_validates_short_question(service: RAGService):
    client = TestClient(create_app(service))
    response = client.post("/api/ask", json={"question": "x"})
    assert response.status_code == 422
