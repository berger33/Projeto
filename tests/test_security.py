"""P3-03: rate limit por IP, token Bearer opcional e /docs desabilitável (G-13, R-21 parte; decisão D5)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.config import ConfigError, Settings
from app.main import create_app
from app.observability import REQUEST_ID_HEADER
from app.rag import RAGService
from app.security import TokenBucketLimiter, client_ip, token_matches

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "segredo-de-teste-com-32-caracteres!!"  # noqa: S105 — valor de teste


@pytest.fixture()
def service() -> RAGService:
    return RAGService(ROOT / "docs", Settings())


def _client(service: RAGService, **overrides: object) -> TestClient:
    security = dataclasses.replace(service.settings, **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(service, security=security))


# ---------------------------------------------------------------------------
# TokenBucketLimiter
# ---------------------------------------------------------------------------


def test_token_bucket_allows_burst_then_refills_over_time() -> None:
    limiter = TokenBucketLimiter(per_minute=60, burst=3)  # 1 token/s
    now = 100.0
    assert [limiter.try_acquire("ip", now=now)[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.try_acquire("ip", now=now)
    assert not allowed and 0.9 < retry_after <= 1.0
    assert limiter.try_acquire("ip", now=now + 1.0)[0] is True  # 1 s depois: 1 token reposto
    assert limiter.try_acquire("outro-ip", now=now)[0] is True  # baldes independentes por chave
    assert len(limiter) == 2


def test_token_bucket_caps_at_burst_and_evicts_idle_keys() -> None:
    limiter = TokenBucketLimiter(per_minute=60, burst=2, max_keys=3)
    now = 0.0
    limiter.try_acquire("a", now=now)
    assert limiter.try_acquire("a", now=now + 1000)[0] and limiter.try_acquire("a", now=now + 1000)[0]
    assert not limiter.try_acquire("a", now=now + 1000)[0]  # nunca acumula além de burst
    for key in ("b", "c"):
        limiter.try_acquire(key, now=now)
    limiter.try_acquire("d", now=now + 5000)  # cheio: chaves ociosas são removidas
    assert len(limiter) <= 3 and "d" in limiter._buckets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_client_ip_ignores_forwarded_header_unless_proxy_is_trusted() -> None:
    scope = {"type": "http", "client": ("10.0.0.9", 1234), "headers": [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.1")]}
    assert client_ip(scope, trust_proxy=False) == "10.0.0.9"
    assert client_ip(scope, trust_proxy=True) == "203.0.113.7"
    assert client_ip({"type": "http", "headers": []}, trust_proxy=True) == "unknown"


def test_token_matches_is_strict_about_scheme_and_value() -> None:
    assert token_matches(f"Bearer {TOKEN}", TOKEN)
    assert token_matches(f"bearer {TOKEN}", TOKEN)
    assert not token_matches(TOKEN, TOKEN)
    assert not token_matches(f"Basic {TOKEN}", TOKEN)
    assert not token_matches(f"Bearer {TOKEN}x", TOKEN)
    assert not token_matches(None, TOKEN) and not token_matches("Bearer ", TOKEN)


# ---------------------------------------------------------------------------
# Defaults (D5): tudo desligado
# ---------------------------------------------------------------------------


def test_security_is_off_by_default(service: RAGService) -> None:
    settings = Settings()
    assert settings.api_token == "" and settings.rate_limit_per_minute == 0 and settings.docs_enabled is True
    with TestClient(create_app(service)) as client:
        assert client.get("/docs").status_code == 200 and client.get("/openapi.json").status_code == 200
        for _ in range(20):
            assert client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).status_code == 200


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def test_api_requires_bearer_token_when_configured(service: RAGService, captured: ListHandler) -> None:
    with _client(service, api_token=TOKEN) as client:
        missing = client.post("/api/ask", json={"question": "Como acompanho meu pedido?"})
        assert missing.status_code == 401
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        body = missing.json()
        assert body["error_code"] == "unauthorized" and body["request_id"] == missing.headers[REQUEST_ID_HEADER]
        assert TOKEN not in missing.text
        wrong = client.post("/api/ask", json={"question": "x y"}, headers={"Authorization": f"Bearer {TOKEN}x"})
        assert wrong.status_code == 401
        ok = client.post(
            "/api/ask", json={"question": "Como acompanho meu pedido?"}, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert ok.status_code == 200
        # Sondas continuam livres.
        assert client.get("/health").status_code == 200 and client.get("/ready").status_code == 200
    events = captured.events("security.unauthorized")
    assert len(events) == 2 and events[0]["path"] == "/api/ask"


def test_short_token_is_rejected_at_boot() -> None:
    with pytest.raises(ConfigError, match="API_TOKEN"):
        Settings(api_token="curto")  # noqa: S106 — valor propositalmente inválido
    assert Settings.from_env({"API_TOKEN": "   "}).api_token == ""
    assert Settings.from_env({"API_TOKEN": TOKEN}).public_dict()["api_token_enabled"] is True
    assert TOKEN not in str(Settings.from_env({"API_TOKEN": TOKEN}).public_dict())


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after_and_standard_error_body(
    service: RAGService, captured: ListHandler
) -> None:
    with _client(service, rate_limit_per_minute=60, rate_limit_burst=3) as client:
        statuses = [
            client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).status_code for _ in range(5)
        ]
        assert statuses == [200, 200, 200, 429, 429]
        blocked = client.post("/api/ask", json={"question": "Como acompanho meu pedido?"})
        assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) >= 1
        body = blocked.json()
        assert set(body) == {"detail", "error_code", "request_id"} and body["error_code"] == "rate_limited"
        assert body["request_id"] == blocked.headers[REQUEST_ID_HEADER]
        # Só POST /api/ask é limitado; sondas e docs não.
        assert client.get("/health").status_code == 200 and client.get("/docs").status_code == 200
    assert captured.events("security.rate_limited")


def test_rate_limit_is_per_client_ip_and_forwarded_header_only_with_trust_proxy(service: RAGService) -> None:
    with _client(service, rate_limit_per_minute=60, rate_limit_burst=1) as client:
        assert client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).status_code == 200
        spoofed = client.post(
            "/api/ask", json={"question": "Como acompanho meu pedido?"}, headers={"X-Forwarded-For": "203.0.113.9"}
        )
        assert spoofed.status_code == 429  # cabeçalho ignorado sem trust_proxy
    with _client(service, rate_limit_per_minute=60, rate_limit_burst=1, trust_proxy=True) as client:
        assert (
            client.post(
                "/api/ask", json={"question": "Como acompanho meu pedido?"}, headers={"X-Forwarded-For": "1.1.1.1"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/ask", json={"question": "Como acompanho meu pedido?"}, headers={"X-Forwarded-For": "2.2.2.2"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/ask", json={"question": "Como acompanho meu pedido?"}, headers={"X-Forwarded-For": "1.1.1.1"}
            ).status_code
            == 429
        )


def test_rate_limit_settings_validation_and_defaults() -> None:
    with pytest.raises(ConfigError, match="RAG_RATE_LIMIT_PER_MINUTE"):
        Settings(rate_limit_per_minute=-1)
    with pytest.raises(ConfigError, match="RAG_RATE_LIMIT_BURST"):
        Settings.from_env({"RAG_RATE_LIMIT_BURST": "0"})
    with pytest.raises(ConfigError, match="RAG_TRUST_PROXY"):
        Settings.from_env({"RAG_TRUST_PROXY": "talvez"})
    settings = Settings.from_env({"RAG_RATE_LIMIT_PER_MINUTE": "30", "RAG_TRUST_PROXY": "yes", "RAG_DOCS_ENABLED": "0"})
    assert settings.rate_limit_per_minute == 30 and settings.rate_limit_burst is None
    assert settings.trust_proxy is True and settings.docs_enabled is False
    public = settings.public_dict()
    assert public["rate_limit_per_minute"] == 30 and public["docs_enabled"] is False


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def test_docs_can_be_disabled(service: RAGService) -> None:
    with _client(service, docs_enabled=False) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).status_code == 200


def test_token_and_rate_limit_compose(service: RAGService) -> None:
    with _client(service, api_token=TOKEN, rate_limit_per_minute=60, rate_limit_burst=1) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        assert client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}).status_code == 401
        assert (
            client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}, headers=headers).status_code == 200
        )
        assert (
            client.post("/api/ask", json={"question": "Como acompanho meu pedido?"}, headers=headers).status_code == 429
        )
        # Requisições não autenticadas não consomem o balde.
        assert client.post("/api/ask", json={"question": "x y"}).status_code == 401
