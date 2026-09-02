"""Observabilidade: logging estruturado, request id por requisição e tempos por etapa.

Somente biblioteca padrão para o logging (JSON por linha). Nenhum dado é enviado para fora do processo.

Eventos emitidos pela aplicação (campo ``event``), todos com ``request_id`` quando dentro de uma requisição:

- ``http.request``      — método, path, status e duração de cada requisição HTTP (``/health`` em DEBUG)
- ``index.built``       — documentos, chunks, dimensão e modelo do índice ao subir
- ``index.error``       — falha ao construir o índice
- ``query.retrieved``   — top-k recuperado (ids + scores) e ids selecionados após os filtros
- ``query.answered``    — status (answered / refused_*), confiança, nº de fontes e tempos por etapa
- ``query.text``        — pergunta e resposta em texto (apenas em DEBUG, por privacidade)
- ``provider.embed``    — chamada de embeddings ao Ollama (nº de textos, tokens, duração)
- ``provider.generate`` — chamada de geração ao Ollama (tokens, done_reason, duração)
- ``provider.error``    — exceção em uma etapa do pipeline (com stack trace)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TypeGuard

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_HANDLER_MARK = "_aurora_log_handler"
_NOISY_LOGGERS = ("httpx", "httpcore")

request_id_var: ContextVar[str | None] = ContextVar("aurora_request_id", default=None)
_http_logger = logging.getLogger("app.http")


# ---------------------------------------------------------------------------
# Request id
# ---------------------------------------------------------------------------


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_request_id() -> str | None:
    """Request id vinculado ao contexto atual (None fora de uma requisição/contexto)."""
    return request_id_var.get()


def is_valid_request_id(value: str | None) -> TypeGuard[str]:
    return value is not None and _REQUEST_ID_RE.match(value) is not None


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Vincula um request id ao contexto atual. Para uso fora do HTTP (CLI, evals, testes)."""
    rid = request_id if is_valid_request_id(request_id) else new_request_id()
    token = request_id_var.set(rid)
    try:
        yield rid
    finally:
        request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Emissão de eventos
# ---------------------------------------------------------------------------


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    /,
    *,
    exc_info: BaseException | bool | None = None,
    **ctx: Any,
) -> None:
    """Emite um evento estruturado. ``ctx`` vira campos de primeiro nível na linha de log."""
    logger.log(level, event, extra={"ctx": ctx}, exc_info=exc_info)


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
        "level": record.levelname,
        "logger": record.name,
        "event": record.getMessage(),
    }
    request_id = request_id_var.get()
    if request_id:
        payload["request_id"] = request_id
    ctx = getattr(record, "ctx", None)
    if isinstance(ctx, dict):
        for key, value in ctx.items():
            payload.setdefault(str(key), value)
    return payload


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro. Valores não serializáveis viram ``str``."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _payload(record)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Formato legível para desenvolvimento: ``ts LEVEL logger event chave=valor ...``."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _payload(record)
        head = f"{payload.pop('ts')} {payload.pop('level'):<7} {payload.pop('logger')} {payload.pop('event')}"
        tail = " ".join(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}" for key, value in payload.items())
        line = f"{head} {tail}".rstrip()
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(level: str | None = None, fmt: str | None = None) -> logging.Handler:
    """Instala (uma única vez) o handler da aplicação no logger raiz e aplica nível/formato.

    ``level`` e ``fmt`` caem para ``LOG_LEVEL`` (padrão INFO) e ``LOG_FORMAT`` (``json`` | ``text``).
    Idempotente: chamadas repetidas apenas reaplicam nível e formato. Handlers de terceiros
    (uvicorn, pytest) são preservados.
    """
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    fmt_name = (fmt or os.getenv("LOG_FORMAT") or "json").strip().lower()
    if level_name not in logging.getLevelNamesMapping():
        raise ValueError(f"LOG_LEVEL inválido: {level_name!r}. Use DEBUG, INFO, WARNING, ERROR ou CRITICAL.")
    if fmt_name not in {"json", "text"}:
        raise ValueError(f"LOG_FORMAT inválido: {fmt_name!r}. Use 'json' ou 'text'.")

    root = logging.getLogger()
    handler = next((h for h in root.handlers if getattr(h, _HANDLER_MARK, False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, _HANDLER_MARK, True)  # marcador para idempotência
        root.addHandler(handler)
    handler.setFormatter(TextFormatter() if fmt_name == "text" else JsonFormatter())
    root.setLevel(level_name)
    for name in _NOISY_LOGGERS:  # httpx loga cada requisição em INFO; os eventos provider.* já cobrem isso
        logging.getLogger(name).setLevel(logging.WARNING)
    return handler


# ---------------------------------------------------------------------------
# Tempos por etapa
# ---------------------------------------------------------------------------


class Timings:
    """Acumula durações em milissegundos por etapa. Etapas repetidas (ex.: lotes) somam."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, milliseconds: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + milliseconds

    def as_dict(self) -> dict[str, float]:
        return {name: round(value, 2) for name, value in self._stages.items()}


def ns_to_ms(value: Any) -> float | None:
    """Converte durações em nanossegundos (formato do Ollama) para ms; None se ausente/inválido."""
    try:
        return round(int(value) / 1_000_000, 2) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Middleware HTTP
# ---------------------------------------------------------------------------


class RequestContextMiddleware:
    """Middleware ASGI puro: gera/propaga ``X-Request-ID``, expõe no contexto e loga cada requisição.

    Um ``X-Request-ID`` recebido só é reaproveitado se for seguro (``[A-Za-z0-9._:-]{1,64}``); caso
    contrário um novo é gerado — evita injeção de conteúdo arbitrário nos logs.
    """

    def __init__(self, app: ASGIApp, *, quiet_paths: frozenset[str] = frozenset({"/health", "/ready"})) -> None:
        self.app = app
        self.quiet_paths = quiet_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if is_valid_request_id(incoming) else new_request_id()
        token = request_id_var.set(request_id)
        status_code = 500
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = str(scope.get("path", ""))
            if status_code >= 500:
                level = logging.WARNING
            elif path in self.quiet_paths:
                level = logging.DEBUG
            else:
                level = logging.INFO
            log_event(
                _http_logger,
                level,
                "http.request",
                method=scope.get("method"),
                path=path,
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
            request_id_var.reset(token)
