"""P2-06: cache de respostas, concorrência controlada ao Ollama e cliente HTTP compartilhado (R-19, G-13 parte)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.cache import AnswerCache, cache_key
from app.config import ConfigError, Settings
from app.domain import AnswerStatus, Generation, RetrievedChunk
from app.embeddings import OllamaEmbeddingProvider
from app.generation import OllamaGenerator
from app.main import create_app
from app.ollama_client import OllamaGate, SharedClient
from app.rag import RAGService

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# AnswerCache
# ---------------------------------------------------------------------------


def test_cache_key_normalizes_question_and_separates_versions() -> None:
    a = cache_key("Qual o prazo de DEVOLUÇÃO?  ", index_version="v1", prompt_version="4", mode="local")
    b = cache_key("qual o prazo de devolucao?", index_version="v1", prompt_version="4", mode="local")
    assert a == b
    assert a != cache_key("qual o prazo de devolucao?", index_version="v2", prompt_version="4", mode="local")
    assert a != cache_key("qual o prazo de devolucao?", index_version="v1", prompt_version="5", mode="local")
    assert a != cache_key("qual o prazo de devolucao?", index_version="v1", prompt_version="4", mode="ollama")


def test_cache_lru_ttl_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("app.cache.time.monotonic", lambda: now[0])
    cache = AnswerCache(max_entries=2, ttl_s=10)
    run = object()
    assert cache.get("a") is None and cache.stats.misses == 1
    cache.put("a", run)  # type: ignore[arg-type]
    cache.put("b", run)  # type: ignore[arg-type]
    assert cache.get("a") is run and cache.stats.hits == 1
    cache.put("c", run)  # type: ignore[arg-type]  # evicta "b" (LRU: "a" foi usado depois de "b")
    assert cache.get("b") is None and cache.stats.evictions == 1
    now[0] += 11
    assert cache.get("a") is None and cache.stats.expirations == 1
    assert cache.stats.as_dict()["size"] == 1 and cache.stats.hit_rate is not None
    assert cache.clear() == 1 and cache.stats.size == 0


def test_cache_disabled_when_zero_entries_or_zero_ttl() -> None:
    for cache in (AnswerCache(max_entries=0), AnswerCache(ttl_s=0)):
        assert not cache.enabled
        cache.put("k", object())  # type: ignore[arg-type]
        assert cache.get("k") is None and cache.stats.size == 0


# ---------------------------------------------------------------------------
# Integração no serviço
# ---------------------------------------------------------------------------


class CountingGenerator:
    mode = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        self.calls += 1
        return Generation(
            text="O prazo para devolução é de 10 dias corridos após o recebimento.", used_sources=(1,), structured=True
        )


@pytest.fixture()
def service() -> RAGService:
    return RAGService(ROOT / "corpus", Settings())


def test_repeated_question_is_served_from_cache_with_fresh_request_id(
    service: RAGService, captured: ListHandler
) -> None:
    generator = CountingGenerator()
    service.generator = generator  # type: ignore[assignment]
    first = service.answer("Qual é o prazo para devolver uma compra?")
    second = service.answer("qual e o prazo para devolver uma compra")  # variação de caixa/acentos
    third = service.answer("Qual é o prazo para devolver uma compra?")
    assert generator.calls == 1
    assert second.answer == first.answer and second.sources == first.sources and second.status is AnswerStatus.ANSWERED
    assert second.request_id != first.request_id and second.timings_ms == {"cache": 0.0}
    assert third.request_id != second.request_id
    cached = captured.events("query.cached")
    assert len(cached) == 2 and cached[-1]["cache"]["hits"] == 2
    assert service.cache.stats.misses == 1


def test_model_refusals_are_not_cached_but_no_context_refusals_are(service: RAGService) -> None:
    class Refusing:
        mode = "stub"
        calls = 0

        def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
            Refusing.calls += 1
            return Generation(text="Não encontrei informação suficiente na documentação.")

    service.generator = Refusing()  # type: ignore[assignment]
    service.answer("Qual é o prazo para devolver uma compra?")
    service.answer("Qual é o prazo para devolver uma compra?")
    assert Refusing.calls == 2  # refused_by_model não entra no cache
    service.answer("Qual é a capital da Austrália?")
    service.answer("Qual é a capital da Austrália?")
    assert service.cache.stats.hits == 1  # refused_no_context é determinístico: cacheado


def test_use_cache_false_and_reload_invalidate(service: RAGService) -> None:
    generator = CountingGenerator()
    service.generator = generator  # type: ignore[assignment]
    service.answer("Como acompanho meu pedido?")
    service.run("Como acompanho meu pedido?", use_cache=False)
    assert generator.calls == 2
    before = service.index_version
    service.reload()
    assert service.index_version == before  # mesmo corpus → mesma versão
    assert service.cache.stats.size == 0  # mas o reload limpa o cache
    service.answer("Como acompanho meu pedido?")
    assert generator.calls == 3


def test_index_version_changes_with_corpus(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("O prazo de devolução é de 10 dias corridos após o recebimento.", encoding="utf-8")
    service = RAGService(docs, Settings(), index_dir=tmp_path / "idx")
    v1 = service.index_version
    (docs / "b.txt").write_text("Aceitamos cartão de crédito e PIX para pagamento.", encoding="utf-8")
    service.reload()
    assert service.index_version != v1


def test_cache_settings_are_validated_and_exposed() -> None:
    assert (
        Settings().cache_max_entries == 256
        and Settings().cache_ttl_s == 600.0
        and Settings().ollama_max_concurrency == 2
    )
    with pytest.raises(ConfigError, match="RAG_CACHE_MAX_ENTRIES"):
        Settings(cache_max_entries=-1)
    with pytest.raises(ConfigError, match="RAG_CACHE_TTL_S"):
        Settings.from_env({"RAG_CACHE_TTL_S": "999999"})
    with pytest.raises(ConfigError, match="OLLAMA_MAX_CONCURRENCY"):
        Settings.from_env({"OLLAMA_MAX_CONCURRENCY": "0"})
    custom = Settings.from_env({"RAG_CACHE_MAX_ENTRIES": "0", "OLLAMA_MAX_CONCURRENCY": "4"})
    assert custom.public_dict()["cache_max_entries"] == 0 and custom.public_dict()["ollama_max_concurrency"] == 4
    service = RAGService(ROOT / "corpus", custom)
    assert not service.cache.enabled and service.gate.max_concurrency == 4


def test_api_answers_are_cached_end_to_end(service: RAGService) -> None:
    generator = CountingGenerator()
    service.generator = generator  # type: ignore[assignment]
    with TestClient(create_app(service)) as client:
        first = client.post("/api/ask", json={"question": "Qual é o prazo para devolver uma compra?"}).json()
        second = client.post("/api/ask", json={"question": "Qual é o prazo para devolver uma compra?"}).json()
    assert generator.calls == 1
    assert first["answer"] == second["answer"] and first["request_id"] != second["request_id"]
    assert second["timings_ms"] == {"cache": 0.0}


# ---------------------------------------------------------------------------
# Concorrência controlada + cliente compartilhado
# ---------------------------------------------------------------------------


def test_gate_limits_concurrent_calls_and_measures_queue_wait() -> None:
    gate = OllamaGate(max_concurrency=2)
    active = 0
    peak = 0
    lock = threading.Lock()
    waits: list[float] = []

    def worker() -> None:
        nonlocal active, peak
        with gate.acquire() as waited:
            waits.append(waited)
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 2
    assert max(waits) >= 40  # ao menos um worker esperou ~1 rodada (50 ms) na fila
    assert gate.waiting == 0 and gate.max_concurrency == 2


def test_gate_is_shared_between_embeddings_and_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f.csv").write_text("pergunta,resposta\nprazo,10 dias.\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
        return httpx.Response(200, json={"message": {"content": "x"}, "done_reason": "stop"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    service = RAGService(tmp_path, Settings(rag_mode="ollama", ollama_max_concurrency=3))
    assert isinstance(service.index.embeddings, OllamaEmbeddingProvider) and isinstance(
        service.generator, OllamaGenerator
    )
    assert service.index.embeddings.gate is service.generator.gate is service.gate
    assert service.gate.max_concurrency == 3


def test_shared_client_is_reused_and_closed(monkeypatch: pytest.MonkeyPatch, captured: ListHandler) -> None:
    created: list[httpx.Client] = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})

    def factory(**kwargs: Any) -> httpx.Client:
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    provider = OllamaEmbeddingProvider("http://o:1", "nomic-embed-text")
    provider.embed_query("a")
    provider.embed_query("b")
    provider.embed_documents(["c", "d"])
    assert len(created) == 1  # uma conexão reutilizada (keep-alive)
    provider.close()
    assert created[0].is_closed
    provider.embed_query("e")  # reabre sob demanda
    assert len(created) == 2
    (event,) = [e for e in captured.events("provider.embed") if e["kind"] == "documents"]
    assert event["queue_wait_ms"] >= 0

    shared = SharedClient(timeout=1.0)
    shared.close()  # fechar sem nunca ter aberto é seguro
    assert shared.get() is shared.get()


def test_lifespan_closes_providers_of_factory_built_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f.csv").write_text("pergunta,resposta\nprazo,O prazo é de 10 dias corridos.\n", encoding="utf-8")
    closed: list[str] = []

    def factory() -> RAGService:
        service = RAGService(tmp_path, Settings())
        monkeypatch.setattr(service, "close", lambda: closed.append("closed"))
        return service

    with TestClient(create_app(service_factory=factory)) as client:
        assert client.get("/health").status_code == 200
    assert closed == ["closed"]
