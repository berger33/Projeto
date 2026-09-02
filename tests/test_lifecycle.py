"""P0-04: Settings validado no boot, RAGService no lifespan, /health vs /ready, erros sem vazamento.

Cobre os achados G-01 (falha de boot virava 500 até em /health), G-02 (503 expunha mensagem interna),
G-03 (lru_cache permitia N construções concorrentes), G-04 (env malformada só falhava na 1ª request)
e G-10 (validação de entrada duplicada/incompleta).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.config import ConfigError, Settings
from app.domain import Generation, RetrievedChunk
from app.embeddings import OllamaEmbeddingProvider
from app.errors import (
    AuroraError,
    InvalidQuestionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ping_ollama,
    provider_call,
)
from app.generation import OllamaGenerator
from app.main import AskRequest, build_service, create_app
from app.observability import REQUEST_ID_HEADER
from app.rag import RAGService

INTERNAL_URL = "http://10.0.0.5:11434"
SECRET = "token=abc"  # noqa: S105 — valor sentinela para detectar vazamento


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
# G-04: Settings validado, mensagens claras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"RAG_MODE": "openai"}, "RAG_MODE='openai'"),
        ({"RAG_TOP_K": "abc"}, "RAG_TOP_K='abc'"),
        ({"RAG_TOP_K": "0"}, "faixa aceita 1..50"),
        ({"RAG_TOP_K": "51"}, "faixa aceita 1..50"),
        ({"RAG_MIN_SCORE": "-1"}, "RAG_MIN_SCORE=-1.0"),
        ({"RAG_MIN_SCORE": "1.5"}, "faixa aceita 0.0..1.0"),
        ({"RAG_MIN_SCORE": "nan"}, "número finito"),
        ({"RAG_MIN_SCORE": "dez"}, "deve ser um número"),
        ({"OLLAMA_EMBED_TIMEOUT_S": "0"}, "OLLAMA_EMBED_TIMEOUT_S"),
        ({"OLLAMA_GENERATE_TIMEOUT_S": "9999"}, "OLLAMA_GENERATE_TIMEOUT_S"),
        ({"RAG_MODE": "ollama", "OLLAMA_BASE_URL": "localhost:11434"}, "OLLAMA_BASE_URL"),
        ({"RAG_MODE": "ollama", "OLLAMA_BASE_URL": "http://host:porta"}, "OLLAMA_BASE_URL"),
    ],
)
def test_settings_rejects_invalid_environment_with_named_variable(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigError, match="Configuração inválida") as info:
        Settings.from_env(env)
    assert fragment in str(info.value)


def test_settings_reports_all_problems_at_once() -> None:
    with pytest.raises(ConfigError) as info:
        Settings(rag_mode="x", retrieval_k=0, min_score=2.0)
    message = str(info.value)
    assert "RAG_MODE" in message and "RAG_TOP_K" in message and "RAG_MIN_SCORE" in message


def test_settings_requires_model_names_in_ollama_mode() -> None:
    with pytest.raises(ConfigError, match="OLLAMA_CHAT_MODEL"):
        Settings(rag_mode="ollama", generation_model="  ")
    with pytest.raises(ConfigError, match="OLLAMA_EMBED_MODEL"):
        Settings(rag_mode="ollama", embedding_model="")
    # Variável em branco no ambiente conta como ausente e cai no default.
    assert Settings.from_env({"RAG_MODE": "ollama", "OLLAMA_CHAT_MODEL": "   "}).generation_model == "qwen3:0.6b"


def test_settings_defaults_and_env_parsing() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.retrieval_k == 5 and settings.min_score == 0.12
    assert settings.embed_timeout_s == 30.0 and settings.generate_timeout_s == 60.0
    assert all(origin == "default" for origin in settings.source.values())

    custom = Settings.from_env(
        {
            "RAG_MODE": " Ollama ",
            "OLLAMA_BASE_URL": "https://ollama.interno:11434/",
            "RAG_TOP_K": " 8 ",
            "RAG_MIN_SCORE": "0.3",
            "OLLAMA_EMBED_TIMEOUT_S": "45",
            "OLLAMA_GENERATE_TIMEOUT_S": "120.5",
            "LOG_LEVEL": "",  # vazio conta como ausente
        }
    )
    assert custom.rag_mode == "ollama" and custom.ollama_base_url == "https://ollama.interno:11434"
    assert custom.retrieval_k == 8 and custom.min_score == 0.3
    assert custom.embed_timeout_s == 45.0 and custom.generate_timeout_s == 120.5
    assert custom.source["RAG_TOP_K"] == "env" and custom.source["OLLAMA_CHAT_MODEL"] == "default"


def test_settings_public_dict_hides_credentials_in_url() -> None:
    settings = Settings(rag_mode="ollama", ollama_base_url="http://user:secret@ollama.interno:11434")
    public = json.dumps(settings.public_dict())
    assert "secret" not in public and "user:" not in public
    assert settings.public_dict()["ollama_host"] == "ollama.interno"
    assert settings.public_dict()["ollama_port"] == 11434
    assert Settings().public_dict()["embedding_model"] == "hash-local"


def test_settings_validation_also_runs_for_direct_construction_and_replace() -> None:
    import dataclasses

    base = Settings()
    with pytest.raises(ConfigError, match="RAG_TOP_K"):
        Settings(retrieval_k=1000)
    with pytest.raises(ConfigError, match="RAG_MIN_SCORE"):
        dataclasses.replace(base, min_score=-0.1)
    # URL do Ollama só é validada quando o modo a usa.
    assert Settings(rag_mode="local", ollama_base_url="not a url").rag_mode == "local"


# ---------------------------------------------------------------------------
# Exceções tipadas dos providers (G-02, parte 1: a origem do detalhe)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected", "code"),
    [
        (httpx.ConnectError("[Errno 111] Connection refused"), ProviderUnavailableError, "provider_unavailable"),
        (httpx.ReadTimeout("timed out"), ProviderTimeoutError, "provider_timeout"),
        (httpx.ConnectTimeout("timed out"), ProviderTimeoutError, "provider_timeout"),
        (ValueError("Expecting value: line 1"), ProviderResponseError, "provider_invalid_response"),
    ],
)
def test_provider_call_translates_httpx_failures(raised: Exception, expected: type[ProviderError], code: str) -> None:
    with pytest.raises(expected) as info, provider_call("embed", f"{INTERNAL_URL}/api/embed"):
        raise raised
    assert info.value.error_code == code
    assert info.value.status_code == 503
    assert INTERNAL_URL in str(info.value)  # detalhe interno preservado para o log...
    assert INTERNAL_URL not in info.value.public_detail  # ...mas nunca no texto público
    assert info.value.__cause__ is raised


def test_provider_call_translates_http_status_error_and_keeps_typed_errors() -> None:
    request = httpx.Request("POST", f"{INTERNAL_URL}/api/generate")
    response = httpx.Response(404, request=request, json={"error": "model 'qwen3:1.7b' not found"})
    with pytest.raises(ProviderUnavailableError, match="HTTP 404") as info, provider_call("generate", str(request.url)):
        response.raise_for_status()
    assert "not found" in str(info.value)

    original = ProviderTimeoutError("já tipado")
    with pytest.raises(ProviderTimeoutError) as info2, provider_call("x", "http://h"):
        raise original
    assert info2.value is original


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_client = httpx.Client

    def patched(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)


def test_ollama_providers_raise_typed_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})  # 1 vetor para 2 textos
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"response": "", "done_reason": "length"})
        return httpx.Response(200, text="not json")

    _mock_client(monkeypatch, handler)
    with pytest.raises(ProviderResponseError, match="1 embedding"):
        OllamaEmbeddingProvider(INTERNAL_URL, "nomic-embed-text").embed_documents(["a", "b"])
    with pytest.raises(ProviderResponseError, match="não devolveu texto"):
        OllamaGenerator(INTERNAL_URL, "qwen3").generate("q", [])
    with pytest.raises(ProviderResponseError):
        ping_ollama(INTERNAL_URL)


def test_ollama_providers_receive_configured_timeouts(docs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    real_client = httpx.Client

    def patched(**kwargs: Any) -> httpx.Client:
        seen.append(float(kwargs["timeout"]))
        return real_client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * 2})),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "Client", patched)
    settings = Settings(rag_mode="ollama", embed_timeout_s=7.0, generate_timeout_s=11.0)
    service = RAGService(docs, settings)
    assert seen == [7.0]
    assert service.generator.timeout == 11.0  # type: ignore[union-attr]


def test_ping_ollama_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}, {"model": "nomic-embed-text:latest"}]})

    _mock_client(monkeypatch, handler)
    assert ping_ollama(INTERNAL_URL) == ["qwen3:1.7b", "nomic-embed-text:latest"]


# ---------------------------------------------------------------------------
# G-01 / G-03: lifespan — uma construção, falha de boot interrompe o servidor
# ---------------------------------------------------------------------------


def test_service_is_built_once_in_lifespan_even_under_concurrency(docs: Path) -> None:
    builds: list[int] = []
    lock = threading.Lock()

    def factory() -> RAGService:
        with lock:
            builds.append(1)
        return RAGService(docs, Settings(rag_mode="local", min_score=0.05, retrieval_k=3))

    app = create_app(service_factory=factory)
    assert builds == []  # nada acontece no import/create_app
    with TestClient(app) as client:
        assert builds == [1]
        statuses: list[int] = []

        def hit() -> None:
            statuses.append(client.post("/api/ask", json={"question": "Qual o prazo de devolução?"}).status_code)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert statuses == [200] * 8
    assert builds == [1]
    assert app.state.rag is None  # limpo no shutdown


def test_startup_fails_fast_on_invalid_configuration(captured: ListHandler) -> None:
    def factory() -> RAGService:
        return RAGService("unused", Settings.from_env({"RAG_TOP_K": "abc"}))

    app = create_app(service_factory=factory)
    with pytest.raises(ConfigError, match="RAG_TOP_K"), TestClient(app):
        pass
    (event,) = captured.events("startup.failed")
    assert event["level"] == "CRITICAL" and event["error_type"] == "ConfigError"


def test_startup_fails_fast_when_ollama_is_unreachable(docs: Path, captured: ListHandler) -> None:
    def factory() -> RAGService:
        return RAGService(docs, Settings(rag_mode="ollama", ollama_base_url="http://127.0.0.1:9"))

    with pytest.raises(ProviderUnavailableError), TestClient(create_app(service_factory=factory)):
        pass
    (event,) = captured.events("startup.failed")
    assert event["error_type"] == "ProviderUnavailableError"
    assert "127.0.0.1:9" in event["error"]  # detalhe vai para o log do operador


def test_startup_fails_fast_when_corpus_is_empty(tmp_path: Path) -> None:
    def factory() -> RAGService:
        return RAGService(tmp_path, Settings(rag_mode="local"))

    with pytest.raises(RuntimeError, match="Nenhum conteúdo"), TestClient(create_app(service_factory=factory)):
        pass


def test_build_service_reads_environment_and_logs_effective_settings(
    monkeypatch: pytest.MonkeyPatch, captured: ListHandler
) -> None:
    monkeypatch.setenv("RAG_MODE", "local")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://user:secret@ollama.interno:11434")
    service = build_service()  # usa o corpus real em docs/
    assert service.settings.retrieval_k == 7 and service.chunk_count > 0
    (event,) = captured.events("settings.loaded")
    assert event["settings"]["retrieval_k"] == 7 and event["source"]["RAG_TOP_K"] == "env"
    assert "secret" not in json.dumps(event)

    monkeypatch.setenv("RAG_MIN_SCORE", "2")
    with pytest.raises(ConfigError, match="RAG_MIN_SCORE"):
        build_service()


def test_default_app_boots_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create_app()`` sem argumentos (como ``app.main:app``) constrói o serviço no lifespan."""
    monkeypatch.setenv("RAG_MODE", "local")
    app = create_app()
    assert app.state.rag is None
    with TestClient(app) as client:
        assert app.state.rag is not None
        assert client.get("/ready").json()["status"] == "ready"
    assert app.state.rag is None


def test_injected_service_skips_factory(service: RAGService) -> None:
    def factory() -> RAGService:
        raise AssertionError("não deve ser chamada quando um serviço é injetado")

    with TestClient(create_app(service, service_factory=factory)) as client:
        assert client.get("/ready").status_code == 200


# ---------------------------------------------------------------------------
# /health (liveness) e /ready (readiness)
# ---------------------------------------------------------------------------


def test_health_is_liveness_only_and_keeps_legacy_fields(service: RAGService) -> None:
    with TestClient(create_app(service)) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"]
    assert body["chunks"] == 2 and body["mode"] == "local-extractive"  # compatibilidade com o contrato anterior


def test_health_answers_even_before_the_service_exists() -> None:
    app = create_app(service_factory=lambda: (_ for _ in ()).throw(RuntimeError("não deve rodar")))
    client = TestClient(app)  # sem `with`: lifespan não executa, simulando o índice ainda não construído
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["error_code"] == "index_not_ready"
    ask = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert ask.status_code == 503 and ask.json()["error_code"] == "index_not_ready"


def test_ready_reports_index_in_local_mode(service: RAGService) -> None:
    with TestClient(create_app(service)) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready" and body["chunks"] == 2
    assert body["checks"] == {"index": {"ok": True, "chunks": 2}}
    assert "ollama" not in body["checks"]
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]


def _ollama_service(docs: Path, monkeypatch: pytest.MonkeyPatch, tags_handler: Any) -> RAGService:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            body = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})
        if request.url.path == "/api/tags":
            return tags_handler(request)
        raise AssertionError(request.url.path)

    _mock_client(monkeypatch, handler)
    return RAGService(
        docs, Settings(rag_mode="ollama", embedding_model="nomic-embed-text", generation_model="qwen3:1.7b")
    )


def test_ready_checks_ollama_and_models_in_ollama_mode(docs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tags = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen3:1.7b"}]}
    service = _ollama_service(docs, monkeypatch, lambda r: httpx.Response(200, json=tags))
    with TestClient(create_app(service)) as client:
        body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["checks"]["ollama"] == {"ok": True, "missing_models": []}


def test_ready_is_503_when_model_is_missing(docs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tags = {"models": [{"name": "nomic-embed-text:latest"}]}
    service = _ollama_service(docs, monkeypatch, lambda r: httpx.Response(200, json=tags))
    with TestClient(create_app(service)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["index"]["ok"] is True
    assert body["checks"]["ollama"] == {"ok": False, "missing_models": ["qwen3:1.7b"]}


def test_ready_is_503_when_ollama_goes_down_after_boot(
    docs: Path, monkeypatch: pytest.MonkeyPatch, captured: ListHandler
) -> None:
    def tags_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = _ollama_service(docs, monkeypatch, tags_down)
    with TestClient(create_app(service)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["ollama"] == {"ok": False, "error_code": "provider_unavailable"}
    assert INTERNAL_URL not in response.text
    (event,) = captured.events("ready.ollama_unreachable")
    assert event["error_code"] == "provider_unavailable"


# ---------------------------------------------------------------------------
# G-02: erros nunca expõem detalhe interno; contrato {detail, error_code, request_id}
# ---------------------------------------------------------------------------


class _FailingGenerator:
    mode = "stub"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (
            ProviderUnavailableError(f"Connection refused to {INTERNAL_URL}/api/generate ({SECRET})"),
            503,
            "provider_unavailable",
        ),
        (ProviderTimeoutError(f"timeout em generate ({INTERNAL_URL})"), 503, "provider_timeout"),
        (ProviderResponseError(f"payload inválido de {INTERNAL_URL}"), 503, "provider_invalid_response"),
        (RuntimeError(f"bug inesperado com {INTERNAL_URL} {SECRET}"), 500, "internal_error"),
        (KeyError("embeddings"), 500, "internal_error"),
    ],
)
def test_pipeline_errors_map_to_generic_contract(
    service: RAGService, captured: ListHandler, exc: Exception, status: int, code: str
) -> None:
    service.generator = _FailingGenerator(exc)  # type: ignore[assignment]
    with TestClient(create_app(service)) as client:
        response = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"})
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"detail", "error_code", "request_id"}
    assert body["error_code"] == code
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert INTERNAL_URL not in response.text and SECRET not in response.text
    assert type(exc).__name__ not in body["detail"]
    # O detalhe completo está no log, correlacionado pelo mesmo request_id.
    (http_error,) = captured.events("http.error")
    assert http_error["status"] == status and http_error["error_code"] == code
    assert http_error["request_id"] == body["request_id"]
    assert INTERNAL_URL in http_error["error"] or type(exc).__name__ in http_error["error"]
    (provider_error,) = captured.events("provider.error")
    assert provider_error["request_id"] == body["request_id"]
    if status == 500:
        assert http_error["level"] == "ERROR" and type(exc).__name__ in http_error["exc_info"]


def test_error_classes_have_stable_public_contract() -> None:
    assert AuroraError.status_code == 500 and AuroraError.error_code == "internal_error"
    for cls in (ProviderError, ProviderUnavailableError, ProviderTimeoutError, ProviderResponseError):
        assert cls.status_code == 503
        assert issubclass(cls, ProviderError)
    assert InvalidQuestionError.status_code == 422
    assert issubclass(InvalidQuestionError, ValueError)  # compatibilidade com chamadores antigos
    assert str(ProviderUnavailableError()) == ProviderUnavailableError.public_detail
    assert InvalidQuestionError("texto do usuário").public_detail == "texto do usuário"


def test_domain_rejects_blank_question_with_typed_error(service: RAGService) -> None:
    with pytest.raises(InvalidQuestionError) as info:
        service.answer("   ")
    assert info.value.status_code == 422
    with pytest.raises(ValueError):  # ainda é um ValueError
        service.answer("")


# ---------------------------------------------------------------------------
# G-10: validação de entrada
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["  ", "x", " a ", "\x00\x00\x00", "ab\x00cd", "pergunta\x1b[31m", "a" * 2001],
)
def test_ask_rejects_invalid_questions_with_422(service: RAGService, question: str) -> None:
    with TestClient(create_app(service)) as client:
        response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 422
    assert response.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("  Qual o prazo de devolução?  ", "Qual o prazo de devolução?"),
        ("prazo\nde devolução", "prazo\nde devolução"),
        ("\tprazo de devolução\t", "prazo de devolução"),
    ],
)
def test_ask_request_strips_and_accepts_line_breaks(raw: str, normalized: str) -> None:
    assert AskRequest(question=raw).question == normalized


def test_ask_accepts_question_with_line_breaks_end_to_end(service: RAGService) -> None:
    with TestClient(create_app(service)) as client:
        response = client.post("/api/ask", json={"question": "Qual o prazo\nde devolução?"})
    assert response.status_code == 200
    assert "10 dias" in response.json()["answer"]
