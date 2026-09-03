"""P1-02: prefixos de tarefa, modelo multilíngue por padrão, batching com retry e validação de dimensão.

Findings: R-05 (crítico: nomic sem prefixos e English-only), G-07 (dimensão não validada), G-08
(indexação em uma única requisição, sem retry).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.conftest import ListHandler

from app.config import ConfigError, Settings
from app.domain import Chunk
from app.embeddings import (
    HashEmbeddingProvider,
    OllamaEmbeddingProvider,
    TaskPrefixes,
    default_prefixes,
)
from app.errors import ProviderResponseError, ProviderTimeoutError, ProviderUnavailableError
from app.rag import RAGService
from app.retrieval import VectorIndex


class FakeOllama:
    """Servidor /api/embed simulado: registra requisições e permite roteirizar falhas."""

    def __init__(self, dimension: int = 3, failures: list[Any] | None = None) -> None:
        self.dimension = dimension
        self.requests: list[dict[str, Any]] = []
        self.failures = list(failures or [])  # consumidos em ordem: int (status HTTP) ou exceção

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, request=request, json={"error": "simulated"})
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1] * self.dimension for _ in body["input"]],
                "prompt_eval_count": len(body["input"]) * 5,
                "total_duration": 1_000_000,
            },
        )


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    server = FakeOllama()
    real_client = httpx.Client

    def patched(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)
    return server


def _provider(**kwargs: Any) -> OllamaEmbeddingProvider:
    kwargs.setdefault("backoff_s", 0.0)
    return OllamaEmbeddingProvider("http://ollama:11434", kwargs.pop("model", "nomic-embed-text-v2-moe"), **kwargs)


# ---------------------------------------------------------------------------
# Prefixos de tarefa (R-05)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "query", "document"),
    [
        ("nomic-embed-text", "search_query: ", "search_document: "),
        ("nomic-embed-text-v2-moe", "search_query: ", "search_document: "),
        ("nomic-embed-text-v2-moe:latest", "search_query: ", "search_document: "),
        ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
        ("embeddinggemma:300m", "task: search result | query: ", "title: none | text: "),
        ("mxbai-embed-large", "Represent this sentence for searching relevant passages: ", ""),
        ("bge-m3", "", ""),
        ("modelo-desconhecido", "", ""),
    ],
)
def test_default_prefixes_follow_model_family(model: str, query: str, document: str) -> None:
    prefixes = default_prefixes(model)
    assert prefixes.query == query and prefixes.document == document


def test_qwen3_embedding_uses_instruction_on_query_only() -> None:
    prefixes = default_prefixes("qwen3-embedding:0.6b")
    assert prefixes.query.startswith("Instruct:") and prefixes.query.endswith("Query: ")
    assert prefixes.document == ""


def test_documents_and_queries_are_sent_with_their_prefixes(fake: FakeOllama) -> None:
    provider = _provider(model="nomic-embed-text-v2-moe")
    provider.embed_documents(["Prazo de devolução: 10 dias.", "Aceitamos PIX."])
    provider.embed_query("qual o prazo de devolução?")
    docs, query = fake.requests
    assert docs["input"] == ["search_document: Prazo de devolução: 10 dias.", "search_document: Aceitamos PIX."]
    assert query["input"] == ["search_query: qual o prazo de devolução?"]
    assert docs["model"] == "nomic-embed-text-v2-moe" and docs["truncate"] is True


def test_prefixes_can_be_overridden_or_disabled(fake: FakeOllama) -> None:
    custom = _provider(prefixes=TaskPrefixes(query="Q: ", document="D: "))
    custom.embed_query("x")
    assert fake.requests[-1]["input"] == ["Q: x"]

    disabled = _provider(prefixes=TaskPrefixes())
    disabled.embed_documents(["texto"])
    assert fake.requests[-1]["input"] == ["texto"]


def test_hash_provider_ignores_prefixes_and_is_symmetric() -> None:
    provider = HashEmbeddingProvider()
    assert provider.embed_query("devolução") == provider.embed_documents(["devolução"])[0]
    assert len(provider.embed_query("x")) == 384


# ---------------------------------------------------------------------------
# Batching e retry (G-08)
# ---------------------------------------------------------------------------


def test_documents_are_embedded_in_batches(fake: FakeOllama, captured: ListHandler) -> None:
    provider = _provider(batch_size=4)
    vectors = provider.embed_documents([f"chunk {i}" for i in range(10)])
    assert len(vectors) == 10
    assert [len(request["input"]) for request in fake.requests] == [4, 4, 2]
    events = captured.events("provider.embed")
    assert [event["texts"] for event in events] == [4, 4, 2]
    assert all(event["kind"] == "documents" and event["dimension"] == 3 for event in events)


def test_batch_size_one_and_empty_input(fake: FakeOllama) -> None:
    assert _provider(batch_size=1).embed_documents([]) == []
    assert fake.requests == []
    _provider(batch_size=1).embed_documents(["a", "b", "c"])
    assert len(fake.requests) == 3


def test_transient_failures_are_retried_with_backoff(
    fake: FakeOllama, captured: ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("app.embeddings.time.sleep", lambda seconds: sleeps.append(seconds))
    fake.failures = [httpx.ReadTimeout("slow"), 503]  # duas falhas transitórias, depois sucesso
    provider = _provider(max_retries=2, backoff_s=0.5)
    vectors = provider.embed_documents(["a"])
    assert len(vectors) == 1 and len(fake.requests) == 3
    assert sleeps == [0.5, 1.0]  # backoff exponencial
    retries = captured.events("provider.retry")
    assert [event["attempt"] for event in retries] == [1, 2]
    assert retries[0]["error_code"] == "provider_timeout" and retries[1]["error_code"] == "provider_unavailable"
    (embed,) = captured.events("provider.embed")
    assert embed["attempts"] == 3


def test_retries_are_exhausted_then_error_propagates(fake: FakeOllama) -> None:
    fake.failures = [httpx.ConnectError("down"), httpx.ConnectError("down"), httpx.ConnectError("down")]
    with pytest.raises(ProviderUnavailableError):
        _provider(max_retries=2).embed_documents(["a"])
    assert len(fake.requests) == 3


def test_non_transient_http_errors_are_not_retried(fake: FakeOllama) -> None:
    fake.failures = [404]  # modelo não instalado
    with pytest.raises(ProviderUnavailableError, match="HTTP 404"):
        _provider(max_retries=3).embed_documents(["a"])
    assert len(fake.requests) == 1

    fake.failures = [httpx.ReadTimeout("slow")]
    with pytest.raises(ProviderTimeoutError):
        _provider(max_retries=0).embed_documents(["a"])


def test_configured_timeout_and_batch_size_reach_the_provider(fake: FakeOllama, tmp_path: Path) -> None:
    (tmp_path / "faq.csv").write_text(
        "pergunta,resposta\n" + "".join(f"q{i},r{i}\n" for i in range(5)), encoding="utf-8"
    )
    settings = Settings(rag_mode="ollama", embed_timeout_s=12.0, embed_batch_size=2)
    service = RAGService(tmp_path, settings)
    assert isinstance(service.index.embeddings, OllamaEmbeddingProvider)
    assert service.index.embeddings.timeout == 12.0 and service.index.embeddings.batch_size == 2
    assert [len(request["input"]) for request in fake.requests] == [2, 2, 1]
    assert service.index.dimension == 3


def test_batch_size_is_validated_in_settings() -> None:
    with pytest.raises(ConfigError, match="OLLAMA_EMBED_BATCH_SIZE"):
        Settings(embed_batch_size=0)
    with pytest.raises(ConfigError, match="OLLAMA_EMBED_BATCH_SIZE"):
        Settings.from_env({"OLLAMA_EMBED_BATCH_SIZE": "1000"})
    assert Settings.from_env({"OLLAMA_EMBED_BATCH_SIZE": "64"}).embed_batch_size == 64


# ---------------------------------------------------------------------------
# Dimensão (G-07)
# ---------------------------------------------------------------------------


class RaggedProvider:
    """Devolve vetores de dimensões diferentes (modelo trocado sem reindexar)."""

    def __init__(self, sizes: list[int], query_size: int = 3) -> None:
        self.sizes = sizes
        self.query_size = query_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * size for size in self.sizes[: len(texts)]]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.query_size


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(id=f"d.csv:r{i}", text=f"texto {i}", source="d.csv", locator={"row": i}) for i in range(n)]


def test_vector_index_rejects_inconsistent_dimensions_at_build() -> None:
    with pytest.raises(RuntimeError, match="dimensões inconsistentes"):
        VectorIndex(_chunks(2), RaggedProvider([2, 3]))
    with pytest.raises(RuntimeError, match="dimensões inconsistentes"):
        VectorIndex(_chunks(1), RaggedProvider([0]))


def test_vector_index_rejects_query_with_other_dimension() -> None:
    index = VectorIndex(_chunks(2), RaggedProvider([3, 3], query_size=4))
    assert index.dimension == 3
    with pytest.raises(RuntimeError, match="dimensão 4"):
        index.search("qualquer coisa")


def test_ollama_provider_rejects_ragged_vectors_from_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.1, 0.2, 0.3]]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(ProviderResponseError, match="dimensão 3; esperado 2"):
        _provider().embed_documents(["a", "b"])


def test_default_embedding_model_is_multilingual() -> None:
    assert Settings().embedding_model == "nomic-embed-text-v2-moe"
    assert default_prefixes(Settings().embedding_model).query == "search_query: "
