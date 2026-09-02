"""Testes de observabilidade (P0-02): logging estruturado, request id e tempos por etapa.

Resolve R-20 (observabilidade ausente) e parte de R-19/G-02 (medição; detalhe de erro vai ao log).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.config import Settings
from app.domain import Generation, RetrievedChunk
from app.embeddings import OllamaEmbeddingProvider
from app.errors import ProviderUnavailableError
from app.generation import OllamaGenerator
from app.main import create_app
from app.observability import (
    REQUEST_ID_HEADER,
    JsonFormatter,
    TextFormatter,
    Timings,
    configure_logging,
    get_request_id,
    is_valid_request_id,
    log_event,
    request_context,
)
from app.rag import RAGService


@pytest.fixture()
def docs(tmp_path: Path) -> Path:
    (tmp_path / "faq.csv").write_text(
        "pergunta,resposta\n"
        "formas de pagamento,Aceitamos cartão de crédito e PIX.\n"
        "devolução,A devolução pode ser solicitada em até 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def service(docs: Path) -> RAGService:
    return RAGService(docs, Settings(rag_mode="local", min_score=0.05, retrieval_k=3))


# ---------------------------------------------------------------------------
# Unidades: formatters, request id, timings, configuração
# ---------------------------------------------------------------------------


def test_json_formatter_emits_one_json_object_per_record_with_context_fields(captured: ListHandler) -> None:
    logger = logging.getLogger("app.test")
    with request_context("req-1"):
        log_event(logger, logging.INFO, "unit.event", answer="ação", count=3, obj=Path("x"))
    (record,) = captured.events("unit.event")
    assert record["level"] == "INFO"
    assert record["logger"] == "app.test"
    assert record["request_id"] == "req-1"
    assert record["answer"] == "ação"  # ensure_ascii=False
    assert record["count"] == 3
    assert record["obj"] == "x"  # objeto não serializável vira str
    assert record["ts"].endswith("+00:00")


def test_json_formatter_includes_stack_trace_on_exception(captured: ListHandler) -> None:
    logger = logging.getLogger("app.test")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        log_event(logger, logging.ERROR, "unit.error", exc_info=exc)
    (record,) = captured.events("unit.error")
    assert "RuntimeError: boom" in record["exc_info"]


def test_text_formatter_is_single_line_key_value() -> None:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "unit.text", None, None)
    record.ctx = {"status": 200, "path": "/x"}
    line = TextFormatter().format(record)
    assert line.count("\n") == 0
    assert "INFO" in line and "unit.text" in line and 'status=200 path="/x"' in line


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("abc-123_x.y:z", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("", False),
        (None, False),
        ("has space", False),
        ('inj"ect{}', False),
        ("line\nbreak", False),
    ],
)
def test_request_id_validation(value: str | None, valid: bool) -> None:
    assert is_valid_request_id(value) is valid


def test_request_context_generates_and_restores_ids() -> None:
    assert get_request_id() is None
    with request_context() as outer:
        assert get_request_id() == outer and len(outer) == 32
        with request_context("inner-id") as inner:
            assert inner == "inner-id" and get_request_id() == "inner-id"
        assert get_request_id() == outer
    assert get_request_id() is None


def test_request_context_replaces_invalid_id() -> None:
    with request_context("not valid!") as rid:
        assert rid != "not valid!" and is_valid_request_id(rid)


def test_timings_accumulate_repeated_stages_and_round() -> None:
    timings = Timings()
    with timings.stage("embed"):
        pass
    timings.add("embed", 1.234567)
    timings.add("search", 2.0)
    result = timings.as_dict()
    assert set(result) == {"embed", "search"}
    assert result["embed"] >= 1.23 and result["search"] == 2.0
    assert all(round(value, 2) == value for value in result.values())


def test_configure_logging_is_idempotent_and_validates_env(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    first = configure_logging(level="INFO", fmt="json")
    second = configure_logging(level="DEBUG", fmt="text")
    assert first is second, "chamadas repetidas não devem empilhar handlers"
    assert len(root.handlers) == before or len(root.handlers) == before + 1
    assert isinstance(second.formatter, TextFormatter)
    assert root.level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        configure_logging(level="LOUD")
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        configure_logging(fmt="xml")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("LOG_FORMAT", "json")
    handler = configure_logging()
    assert root.level == logging.WARNING and isinstance(handler.formatter, JsonFormatter)
    configure_logging(level="WARNING", fmt="json")  # deixa o estado previsível para outros testes


# ---------------------------------------------------------------------------
# Pipeline: eventos e tempos por etapa
# ---------------------------------------------------------------------------


def test_index_built_event_describes_the_index(docs: Path, captured: ListHandler) -> None:
    RAGService(docs, Settings(rag_mode="local"))
    (event,) = captured.events("index.built")
    assert event["chunks"] == 2 and event["documents"] == 1 and event["dimension"] == 384
    assert event["mode"] == "local-extractive" and event["embedding_model"] == "hash-local"
    assert event["duration_ms"] >= 0


def test_index_error_event_is_logged_and_reraised(tmp_path: Path, captured: ListHandler) -> None:
    with pytest.raises(RuntimeError):
        RAGService(tmp_path, Settings(rag_mode="local"))  # diretório sem PDF/CSV
    (event,) = captured.events("index.error")
    assert event["error_type"] == "IngestError" and "exc_info" in event


def test_answer_emits_retrieved_and_answered_events_with_stage_timings(
    service: RAGService, captured: ListHandler
) -> None:
    result = service.answer("Quais formas de pagamento são aceitas?")
    (retrieved,) = captured.events("query.retrieved")
    (answered,) = captured.events("query.answered")
    assert retrieved["k"] == 3 and len(retrieved["candidates"]) == 2
    assert all({"id", "score"} <= set(item) for item in retrieved["candidates"])
    assert retrieved["selected"] and retrieved["selected"][0] == "faq.csv:r2"
    assert answered["status"] == "answered" and answered["sources"] == 1
    assert set(answered["timings_ms"]) == {"retrieve", "filter", "generate", "verify"}
    assert answered["total_ms"] >= sum(answered["timings_ms"].values()) * 0.5
    assert result.timings_ms == answered["timings_ms"]
    # Fora de uma requisição HTTP, a resposta ganha o próprio request_id e ele aparece nos eventos.
    assert result.request_id and retrieved["request_id"] == answered["request_id"] == result.request_id


def test_refusal_without_context_is_logged_with_status(service: RAGService, captured: ListHandler) -> None:
    result = service.answer("Qual é a capital da Austrália?")
    (answered,) = captured.events("query.answered")
    assert answered["status"] == "refused_no_context" and answered["sources"] == 0
    assert "generate" not in result.timings_ms and "retrieve" in result.timings_ms


def test_refusal_by_model_is_logged_with_status(service: RAGService, captured: ListHandler) -> None:
    class RefusingGenerator:
        mode = "stub"

        def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
            return Generation(text="Não encontrei informação suficiente na documentação.")

    service.generator = RefusingGenerator()  # type: ignore[assignment]
    service.answer("Quais formas de pagamento são aceitas?")
    (answered,) = captured.events("query.answered")
    assert answered["status"] == "refused_by_model" and answered["sources"] == 0
    assert answered["refusal_reason"] == "pattern"
    (refused,) = captured.events("answer.refused")
    assert refused["reason"] == "pattern" and refused["matched_pattern"]


def test_question_and_answer_text_only_logged_at_debug(service: RAGService, captured: ListHandler) -> None:
    logging.getLogger("app.rag").setLevel(logging.INFO)
    try:
        service.answer("Quais formas de pagamento são aceitas?")
        assert captured.events("query.text") == []
    finally:
        logging.getLogger("app.rag").setLevel(logging.NOTSET)
    service.answer("Quais formas de pagamento são aceitas?")
    (text,) = captured.events("query.text")
    assert text["question"].startswith("Quais formas") and "PIX" in text["answer"]


def test_provider_failure_is_logged_with_stage_and_reraised(service: RAGService, captured: ListHandler) -> None:
    class BrokenGenerator:
        mode = "stub"

        def generate(self, question: str, context: list[RetrievedChunk]) -> str:
            raise httpx.ConnectError("connection refused to http://10.0.0.5:11434")

    service.generator = BrokenGenerator()  # type: ignore[assignment]
    with pytest.raises(httpx.ConnectError):
        service.answer("Quais formas de pagamento são aceitas?")
    (error,) = captured.events("provider.error")
    assert error["error_type"] == "ConnectError" and error["stage"] == "generate"
    assert "10.0.0.5" in error["exc_info"], "o detalhe fica no log do operador"


# ---------------------------------------------------------------------------
# Providers Ollama (transporte simulado): eventos com métricas do servidor
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    calls = {"embed": 0, "generate": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            calls["embed"] += 1
            return httpx.Response(
                200,
                json={
                    "embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]],
                    "prompt_eval_count": 12,
                    "total_duration": 5_000_000,
                    "load_duration": 1_000_000,
                },
            )
        calls["generate"] += 1
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "O prazo é de 10 dias corridos."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 200,
                "eval_count": 12,
                "total_duration": 800_000_000,
                "load_duration": 10_000_000,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 250_000_000,
            },
        )

    real_client = httpx.Client

    def patched_client(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", patched_client)
    yield calls


def test_ollama_embed_event_reports_server_metrics(mock_ollama: dict[str, int], captured: ListHandler) -> None:
    vectors = OllamaEmbeddingProvider("http://ollama:11434", "nomic-embed-text").embed_documents(["a", "bb"])
    assert len(vectors) == 2 and mock_ollama["embed"] == 1
    (event,) = captured.events("provider.embed")
    # chars conta o texto enviado, já com o prefixo "search_document: " (17 chars) em cada item.
    assert event["model"] == "nomic-embed-text" and event["texts"] == 2 and event["chars"] == 3 + 2 * 17
    assert event["dimension"] == 3 and event["prompt_tokens"] == 12
    assert event["ollama_total_ms"] == 5.0 and event["ollama_load_ms"] == 1.0
    assert event["duration_ms"] >= 0


def test_ollama_generate_event_reports_tokens_and_done_reason(
    mock_ollama: dict[str, int], captured: ListHandler
) -> None:
    from app.domain import Chunk

    context = [RetrievedChunk(Chunk("faq.pdf:p1:c1", "O prazo é de 10 dias.", "faq.pdf", {"page": 1}), 0.9)]
    generation = OllamaGenerator("http://ollama:11434", "qwen3:1.7b").generate("Qual o prazo?", context)
    assert generation.text.startswith("O prazo") and mock_ollama["generate"] == 1
    (event,) = captured.events("provider.generate")
    assert event["model"] == "qwen3:1.7b" and event["context_chunks"] == 1 and event["prompt_version"]
    assert event["prompt_tokens"] == 200 and event["completion_tokens"] == 12 and event["done_reason"] == "stop"
    assert event["ollama_total_ms"] == 800.0 and event["ollama_prompt_eval_ms"] == 500.0
    assert event["prompt_chars"] > 0 and event["answer_chars"] == len(generation.text)
    assert event["structured"] is False  # o mock devolve texto puro, não o JSON pedido


# ---------------------------------------------------------------------------
# HTTP: middleware de request id e log de requisição
# ---------------------------------------------------------------------------


def test_http_generates_request_id_and_returns_it_in_header_and_body(
    service: RAGService, captured: ListHandler
) -> None:
    client = TestClient(create_app(service))
    response = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert response.status_code == 200
    rid = response.headers[REQUEST_ID_HEADER]
    assert is_valid_request_id(rid) and len(rid) == 32
    body = response.json()
    assert body["request_id"] == rid
    assert set(body["timings_ms"]) == {"retrieve", "filter", "generate", "verify"}
    # O mesmo id correlaciona todos os eventos da requisição.
    for name in ("query.retrieved", "query.answered", "http.request"):
        (event,) = captured.events(name)
        assert event["request_id"] == rid, name
    (http,) = captured.events("http.request")
    assert http["method"] == "POST" and http["path"] == "/api/ask" and http["status"] == 200
    assert http["duration_ms"] >= 0


def test_http_reuses_valid_incoming_request_id(service: RAGService, captured: ListHandler) -> None:
    client = TestClient(create_app(service))
    response = client.post(
        "/api/ask", json={"question": "Qual o prazo de devolução?"}, headers={REQUEST_ID_HEADER: "client-42"}
    )
    assert response.headers[REQUEST_ID_HEADER] == "client-42"
    assert response.json()["request_id"] == "client-42"
    assert captured.events("query.answered")[0]["request_id"] == "client-42"


def test_http_rejects_unsafe_incoming_request_id(service: RAGService) -> None:
    client = TestClient(create_app(service))
    response = client.get("/health", headers={REQUEST_ID_HEADER: 'bad id "}{'})
    rid = response.headers[REQUEST_ID_HEADER]
    assert rid != 'bad id "}{' and is_valid_request_id(rid)


def test_health_is_logged_at_debug_and_errors_at_warning(
    service: RAGService, captured: ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")  # create_app() aplica LOG_LEVEL; /health só aparece em DEBUG
    client = TestClient(create_app(service))
    client.get("/health")
    (health,) = captured.events("http.request")
    assert health["level"] == "DEBUG" and health["path"] == "/health"

    class BrokenGenerator:
        mode = "stub"

        def generate(self, question: str, context: list[RetrievedChunk]) -> str:
            raise ProviderUnavailableError("connection refused to http://10.0.0.5:11434/api/generate")

    service.generator = BrokenGenerator()  # type: ignore[assignment]
    response = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert response.status_code == 503
    assert response.headers[REQUEST_ID_HEADER]
    http_events = captured.events("http.request")
    assert http_events[-1]["status"] == 503 and http_events[-1]["level"] == "WARNING"
    (error,) = captured.events("provider.error")
    assert error["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_validation_errors_also_carry_request_id(service: RAGService) -> None:
    client = TestClient(create_app(service))
    response = client.post("/api/ask", json={"question": "x"})
    assert response.status_code == 422 and is_valid_request_id(response.headers[REQUEST_ID_HEADER])


def test_answer_contract_remains_backward_compatible(service: RAGService) -> None:
    """Campos antigos preservados; novos são aditivos (decisão D10)."""
    client = TestClient(create_app(service))
    body = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"}).json()
    assert {"answer", "sources", "confidence", "mode"} <= set(body)
    assert {"request_id", "timings_ms"} <= set(body)
