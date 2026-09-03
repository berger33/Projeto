"""P2-07: lacunas remanescentes da suíte do pipeline (R-22, G-20).

Cobre o que os itens anteriores não fecharam: PDF real multi-página gerado em teste (texto extraível,
cabeçalho/rodapé repetidos, seções), contrato do Ollama em casos de borda (404 de modelo, JSON inválido,
dimensão inconsistente entre lotes, resposta cortada), mutantes que sobreviviam (sinal do hash) e o
comportamento ponta a ponta sobre um corpus PDF de verdade.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler, write_pdf

from app.config import Settings
from app.documents import load_corpus
from app.embeddings import HashEmbeddingProvider, OllamaEmbeddingProvider
from app.errors import ProviderResponseError, ProviderUnavailableError
from app.generation import OllamaGenerator
from app.main import create_app
from app.rag import RAGService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def pdf_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    write_pdf(
        docs / "politica.pdf",
        [
            [
                "AURORA MODA ONLINE",
                "Politica de Trocas",
                "1. Prazo para troca",
                "O cliente pode solicitar a troca em ate 30 dias corridos apos o recebimento do",
                "pedido, informando o numero do pedido ao suporte.",
                "2. Condicoes do produto",
                "O produto deve estar sem sinais de uso, com etiqueta e embalagem original.",
                "Documento ficticio para fins academicos.",
            ],
            [
                "AURORA MODA ONLINE",
                "3. Custos",
                "A primeira troca e gratuita; a partir da segunda o frete de retorno e cobrado do cliente.",
                "4. Contato",
                "Solicitacoes de troca devem ser enviadas para trocas@auroramoda.exemplo.",
                "Documento ficticio para fins academicos.",
            ],
        ],
    )
    return docs


# ---------------------------------------------------------------------------
# PDF real: extração, boilerplate entre páginas, seções, rastreabilidade de página
# ---------------------------------------------------------------------------


def test_real_pdf_pages_sections_and_boilerplate(pdf_corpus: Path) -> None:
    chunks, report = load_corpus(pdf_corpus)
    (entry,) = report.files
    assert entry.kind == "pdf" and entry.pages == 2 and entry.empty_pages == 0 and entry.chunks == len(chunks) >= 3
    pages = {chunk.locator["page"] for chunk in chunks}
    assert pages == {1, 2}
    for chunk in chunks:
        assert "AURORA MODA ONLINE" not in chunk.text and "Documento ficticio" not in chunk.text
        assert chunk.section and chunk.char_start is not None and chunk.token_estimate
    sections = [chunk.section for chunk in chunks]
    assert any("1. Prazo para troca" in section for section in sections)
    assert any("4. Contato" in section for section in sections)
    # Quebra visual de linha dentro da frase foi unida ("recebimento do\npedido" → "recebimento do pedido").
    assert any("recebimento do pedido" in chunk.text for chunk in chunks)


def test_end_to_end_over_real_pdf_answers_with_page_and_excerpt(pdf_corpus: Path, tmp_path: Path) -> None:
    service = RAGService(pdf_corpus, Settings(), index_dir=tmp_path / "idx")
    result = service.answer("Qual o prazo para solicitar troca?")
    assert result.status == "answered" and "30 dias" in result.answer
    (source, *_) = result.sources
    assert source.document == "politica.pdf" and source.page == 1 and source.chunk_id.startswith("politica.pdf:p1")
    assert source.excerpt and "30 dias" in source.excerpt and source.section and "Prazo" in source.section

    contact = service.answer("Para qual e-mail envio a solicitação de troca?")
    assert contact.status == "answered" and "trocas@auroramoda.exemplo" in contact.answer
    assert contact.sources[0].page == 2

    assert service.answer("Qual é a capital da Austrália?").status == "refused_no_context"


def test_api_over_real_pdf_reports_page_in_sources(pdf_corpus: Path, tmp_path: Path) -> None:
    service = RAGService(pdf_corpus, Settings(), index_dir=tmp_path / "idx")
    with TestClient(create_app(service)) as client:
        body = client.post("/api/ask", json={"question": "A primeira troca é gratuita?"}).json()
    assert body["status"] == "answered" and body["sources"][0]["page"] == 2
    assert body["sources"][0]["row"] is None


# ---------------------------------------------------------------------------
# Contrato Ollama: bordas ainda não cobertas
# ---------------------------------------------------------------------------


def _mock(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def test_embed_dimension_mismatch_between_batches_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        width = 3 if calls["n"] == 1 else 4  # segundo lote com outra dimensão (modelo trocado no meio)
        return httpx.Response(200, json={"embeddings": [[0.1] * width for _ in body["input"]]})

    _mock(monkeypatch, handler)
    with pytest.raises(ProviderResponseError, match="dimensão 4; esperado 3"):
        OllamaEmbeddingProvider("http://o:1", "m", batch_size=2, backoff_s=0).embed_documents(["a", "b", "c"])


def test_missing_model_404_is_reported_with_server_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:1.7b' not found, try pulling it first"})

    _mock(monkeypatch, handler)
    with pytest.raises(ProviderUnavailableError) as info:
        OllamaGenerator("http://o:1", "qwen3:1.7b").generate("q", [])
    assert "HTTP 404" in str(info.value) and "try pulling it first" in str(info.value)
    assert info.value.public_detail == ProviderUnavailableError.public_detail


def test_invalid_json_from_server_is_provider_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock(monkeypatch, lambda request: httpx.Response(200, text="<html>bad gateway</html>"))
    with pytest.raises(ProviderResponseError):
        OllamaGenerator("http://o:1", "m").generate("q", [])
    with pytest.raises(ProviderResponseError):
        OllamaEmbeddingProvider("http://o:1", "m").embed_query("q")


def test_ollama_service_end_to_end_with_structured_answer_and_503_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured: ListHandler
) -> None:
    (tmp_path / "faq.csv").write_text(
        "pergunta,resposta\nprazo,O prazo de devolução é de 10 dias corridos após o recebimento.\n", encoding="utf-8"
    )
    state = {"down": False}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            # Vetor determinístico por texto: pergunta e chunk sobre "prazo" ficam próximos.
            vectors = [[1.0, 0.2] if "prazo" in text.lower() else [0.0, 1.0] for text in body["input"]]
            return httpx.Response(200, json={"embeddings": vectors})
        if state["down"]:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"answer": "O prazo é de 10 dias corridos [1].", "grounded": true, "used_sources": [1]}'
                },
                "done_reason": "stop",
                "prompt_eval_count": 120,
                "eval_count": 15,
            },
        )

    _mock(monkeypatch, handler)
    service = RAGService(tmp_path, Settings(rag_mode="ollama"), index_dir="")
    with TestClient(create_app(service)) as client:
        ok = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
        assert ok.status_code == 200
        body = ok.json()
        assert body["status"] == "answered" and body["answer"] == "O prazo é de 10 dias corridos."
        assert body["sources"][0]["inferred"] is False and body["mode"] == "ollama"
        state["down"] = True
        service.cache.clear()
        down = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert down.status_code == 503 and down.json()["error_code"] == "provider_unavailable"
    generate = captured.events("provider.generate")[0]
    assert generate["structured"] is True and generate["prompt_tokens"] == 120


# ---------------------------------------------------------------------------
# Mutantes remanescentes (G-20)
# ---------------------------------------------------------------------------


def test_hash_embedding_sign_is_documented_as_non_discriminative() -> None:
    """R-07 (Fase 2): o bit de sinal do hash é derivado do mesmo digest que o bucket, então não reduz o
    cosseno entre textos disjuntos de forma mensurável — o mutante "remover o sinal" sobrevive por
    construção. Este teste registra a limitação em vez de fingir uma garantia: o cosseno médio entre
    textos sem palavras em comum fica na mesma faixa com ou sem sinal (~0,15 a 60 tokens/384 dims),
    e é por isso que o perfil `local` só serve como harness de teste, não como retrieval semântico.
    """
    import hashlib
    import math
    import random

    def vector(tokens: list[str], *, signed: bool) -> list[float]:
        values = [0.0] * 384
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            values[number % 384] += (-1.0 if (number >> 1) & 1 else 1.0) if signed else 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    rng = random.Random(0)  # noqa: S311 — gerador de casos de teste
    vocabulary = [f"w{i}" for i in range(5000)]
    means = {}
    for signed in (True, False):
        cosines = []
        for _ in range(100):
            left, right = rng.sample(vocabulary, 60), rng.sample(vocabulary, 60)
            cosines.append(
                sum(a * b for a, b in zip(vector(left, signed=signed), vector(right, signed=signed), strict=True))
            )
        means[signed] = sum(cosines) / len(cosines)
    assert abs(means[True] - means[False]) < 0.02  # o sinal não discrimina
    provider = HashEmbeddingProvider()
    assert any(value < 0 for value in provider.embed_query("prazo devolução pagamento"))  # mas existe no código


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider()
    vector = provider.embed_query("devolução prazo")
    assert vector == provider.embed_query("devolução prazo")
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6
    assert provider.embed_query("") == [0.0] * 384
