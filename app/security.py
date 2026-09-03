"""Rate limit por IP e autenticação por token opcional (Fase 2, G-13/R-21 parte; decisão D5).

Ambos desligados por padrão — a API é tratada como não pública (D5). Ligados por variáveis de ambiente:

- ``RAG_RATE_LIMIT_PER_MINUTE`` (> 0 liga): *token bucket* por IP em memória para ``POST /api/ask``;
  excesso responde ``429`` com ``Retry-After`` e o contrato de erro padrão (``error_code=rate_limited``).
  A capacidade do balde é ``RAG_RATE_LIMIT_BURST`` (padrão = limite/minuto). Com vários workers cada
  processo tem o próprio balde (limite efetivo = N x valor) — documentado; um limitador distribuído
  fica atrás de proxy/gateway.
- ``API_TOKEN`` (não vazio liga): ``Authorization: Bearer <token>`` obrigatório em ``/api/*``;
  comparação em tempo constante; ``401`` com ``WWW-Authenticate``. ``/health`` e ``/ready`` continuam
  livres (sondas de orquestrador).
- ``RAG_DOCS_ENABLED=false`` desliga ``/docs``, ``/redoc`` e ``/openapi.json``.

O IP do cliente vem de ``X-Forwarded-For`` **somente** quando ``RAG_TRUST_PROXY=true`` (atrás de um
proxy reverso confiável); caso contrário usa o peer da conexão, para que o cabeçalho não seja usado
para escapar do limite.
"""

from __future__ import annotations

import hmac
import logging
import math
import threading
import time
from dataclasses import dataclass, field

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import AuroraError
from .observability import get_request_id, log_event

logger = logging.getLogger(__name__)


class RateLimitedError(AuroraError):
    status_code = 429
    error_code = "rate_limited"
    public_detail = "Muitas requisições. Aguarde alguns segundos e tente novamente."


class UnauthorizedError(AuroraError):
    status_code = 401
    error_code = "unauthorized"
    public_detail = "Credencial ausente ou inválida."


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class TokenBucketLimiter:
    """``per_minute`` tokens repostos continuamente; ``burst`` = capacidade. Thread-safe."""

    per_minute: float
    burst: float
    _buckets: dict[str, _Bucket] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    max_keys: int = 10_000

    @property
    def rate_per_second(self) -> float:
        return self.per_minute / 60.0

    def try_acquire(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """``(permitido, segundos até o próximo token)``."""
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    self._evict(now)
                bucket = _Bucket(tokens=self.burst, updated=now)
                self._buckets[key] = bucket
            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate_per_second)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            retry_after = (1.0 - bucket.tokens) / self.rate_per_second if self.rate_per_second else 60.0
            return False, retry_after

    def _evict(self, now: float) -> None:
        # Remove baldes cheios há mais de um período completo (clientes inativos).
        idle = self.burst / self.rate_per_second if self.rate_per_second else 60.0
        for key in [key for key, bucket in self._buckets.items() if now - bucket.updated > idle]:
            del self._buckets[key]
        if len(self._buckets) >= self.max_keys:  # ainda cheio: descarta o mais antigo
            oldest = min(self._buckets, key=lambda key: self._buckets[key].updated)
            del self._buckets[oldest]

    def __len__(self) -> int:
        return len(self._buckets)


def client_ip(scope: Scope, *, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = Headers(scope=scope).get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip() or "unknown"
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def token_matches(header_value: str | None, expected: str) -> bool:
    if not header_value:
        return False
    scheme, _, credential = header_value.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        return False
    return hmac.compare_digest(credential.strip().encode("utf-8"), expected.encode("utf-8"))


class SecurityMiddleware:
    """Aplica token (em ``/api/*``) e rate limit (em ``POST /api/ask``) antes de chegar às rotas.

    Middlewares de aplicação rodam **fora** do ``ExceptionMiddleware`` do Starlette, então este
    middleware responde por conta própria, com o mesmo contrato ``{detail, error_code, request_id}``
    do handler global (mais ``Retry-After`` / ``WWW-Authenticate``).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_token: str = "",
        limiter: TokenBucketLimiter | None = None,
        trust_proxy: bool = False,
        protected_prefix: str = "/api/",
        limited_paths: frozenset[str] = frozenset({"/api/ask"}),
    ) -> None:
        self.app = app
        self.api_token = api_token
        self.limiter = limiter
        self.trust_proxy = trust_proxy
        self.protected_prefix = protected_prefix
        self.limited_paths = limited_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        if path.startswith(self.protected_prefix):
            ip = client_ip(scope, trust_proxy=self.trust_proxy)
            if self.api_token and not token_matches(Headers(scope=scope).get("authorization"), self.api_token):
                log_event(logger, logging.WARNING, "security.unauthorized", path=path, ip=ip)
                await self._reject(scope, receive, send, UnauthorizedError(), {"WWW-Authenticate": "Bearer"})
                return
            if self.limiter is not None and path in self.limited_paths and scope.get("method") == "POST":
                allowed, retry_after = self.limiter.try_acquire(ip)
                if not allowed:
                    log_event(
                        logger,
                        logging.WARNING,
                        "security.rate_limited",
                        path=path,
                        ip=ip,
                        retry_after_s=round(retry_after, 2),
                    )
                    headers = {"Retry-After": str(max(1, math.ceil(retry_after)))}
                    await self._reject(scope, receive, send, RateLimitedError(), headers)
                    return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, error: AuroraError, headers: dict[str, str]) -> None:
        body = {"detail": error.public_detail, "error_code": error.error_code, "request_id": get_request_id()}
        response = JSONResponse(status_code=error.status_code, content=body, headers=headers)
        await response(scope, receive, send)
