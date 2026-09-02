from __future__ import annotations

import logging
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, field_validator

from .config import Settings
from .domain import AnswerStatus, Confidence
from .errors import AuroraError, IndexNotReadyError
from .observability import RequestContextMiddleware, configure_logging, get_request_id, log_event
from .rag import RAGService
from .security import SecurityMiddleware, TokenBucketLimiter

BASE = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

ServiceFactory = Callable[[], RAGService]


# ---------------------------------------------------------------------------
# Contratos HTTP
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)]

    @field_validator("question")
    @classmethod
    def _reject_control_characters(cls, value: str) -> str:
        # Categoria Unicode "Cc" = caracteres de controle (NUL, ESC, ...). Quebras de linha e tab são aceitas.
        if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in value):
            raise ValueError("A pergunta contém caracteres de controle não permitidos.")
        return value


class SourceResponse(BaseModel):
    document: str
    page: int | None = None
    row: int | None = None
    # Aditivos (D10): rastreabilidade até o chunk e o trecho que sustentou a resposta.
    chunk_id: str | None = None
    score: float | None = None
    section: str | None = None
    excerpt: str | None = None
    inferred: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    confidence: Confidence
    mode: str
    request_id: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    # Aditivos (D10): estado da resposta e motivo da recusa, para clientes que queiram distinguir os casos.
    status: AnswerStatus = AnswerStatus.ANSWERED
    refusal_reason: str | None = None


class ErrorResponse(BaseModel):
    """Corpo padrão de erro. ``detail`` é sempre genérico; o detalhe fica no log com o mesmo ``request_id``."""

    detail: str
    error_code: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------


def build_service() -> RAGService:
    """Constrói o serviço a partir do ambiente. ``ConfigError`` ou falha do índice interrompem o boot."""
    settings = Settings.from_env()
    log_event(logger, logging.INFO, "settings.loaded", settings=settings.public_dict(), source=settings.source)
    return RAGService(BASE / "docs", settings)


def create_app(
    service: RAGService | None = None,
    *,
    service_factory: ServiceFactory | None = None,
    security: Settings | None = None,
) -> FastAPI:
    """Cria a aplicação.

    ``service`` (testes) injeta um serviço já construído. Caso contrário ``service_factory`` (padrão
    ``build_service``) roda **uma única vez** no ``lifespan``, antes de o servidor aceitar conexões.
    Se falhar, o processo não sobe: o erro é logado (``startup.failed``) e propagado para o uvicorn
    encerrar com código de saída ≠ 0 (Fase 2, G-01/G-03).

    ``security`` define token/rate limit/docs (padrão: ``service.settings`` quando injetado, senão o
    ambiente). É lida na criação da app — antes do lifespan — porque middlewares e rotas de docs
    precisam existir desde o início.
    """
    configure_logging()
    factory = service_factory or build_service
    security_settings = security or (service.settings if service is not None else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.rag = service
        else:
            try:
                app.state.rag = factory()
            except Exception as exc:
                log_event(logger, logging.CRITICAL, "startup.failed", error_type=type(exc).__name__, error=str(exc))
                raise
        yield
        rag = getattr(app.state, "rag", None)
        if rag is not None and service is None:
            rag.close()
        app.state.rag = None

    app = FastAPI(
        title="Aurora Document RAG API",
        version="2.1.0",
        description="RAG documental com retrieval vetorial, geração local opcional e citações.",
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
        docs_url="/docs" if security_settings.docs_enabled else None,
        redoc_url="/redoc" if security_settings.docs_enabled else None,
        openapi_url="/openapi.json" if security_settings.docs_enabled else None,
    )
    app.state.rag = service  # serviço injetado fica disponível mesmo sem o lifespan (TestClient sem `with`)
    # Ordem: o último add_middleware é o mais externo → RequestContext envolve Security (request_id nos 401/429).
    limiter = (
        TokenBucketLimiter(
            per_minute=security_settings.rate_limit_per_minute,
            burst=security_settings.rate_limit_burst or security_settings.rate_limit_per_minute,
        )
        if security_settings.rate_limit_per_minute > 0
        else None
    )
    if limiter is not None or security_settings.api_token:
        app.add_middleware(
            SecurityMiddleware,
            api_token=security_settings.api_token,
            limiter=limiter,
            trust_proxy=security_settings.trust_proxy,
        )
    app.state.limiter = limiter
    app.add_middleware(RequestContextMiddleware)

    def get_service(request: Request) -> RAGService:
        rag = getattr(request.app.state, "rag", None)
        if rag is None:
            raise IndexNotReadyError("RAGService ausente em app.state (lifespan não executou ou falhou).")
        return rag  # type: ignore[no-any-return]

    # ----- erros: nunca expõem mensagem interna -----

    @app.exception_handler(AuroraError)
    async def aurora_error_handler(request: Request, exc: AuroraError) -> JSONResponse:
        # A mensagem interna (URL do Ollama, erro de rede, stack) fica só aqui, correlacionada pelo request_id.
        internal = exc.error_code == AuroraError.error_code
        log_event(
            logger,
            logging.ERROR if internal else (logging.WARNING if exc.status_code >= 500 else logging.INFO),
            "http.error",
            exc_info=exc if internal else None,
            status=exc.status_code,
            error_code=exc.error_code,
            error_type=type(exc.__cause__ or exc).__name__,
            error=str(exc),
        )
        body = ErrorResponse(detail=exc.public_detail, error_code=exc.error_code, request_id=get_request_id())
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    # ----- rotas -----

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        """Liveness: o processo responde. Nunca consulta o Ollama (isso é papel de ``/ready``).

        ``chunks`` e ``mode`` são mantidos por compatibilidade com o contrato anterior.
        """
        body: dict[str, object] = {"status": "ok", "version": app.version}
        rag = getattr(request.app.state, "rag", None)
        if rag is not None:
            body.update(chunks=rag.chunk_count, mode=rag.generator.mode)
        return body

    @app.get("/ready", responses={503: {"model": ErrorResponse}})
    def ready(request: Request) -> JSONResponse:
        """Readiness: índice construído (chunks > 0) e, em modo Ollama, servidor e modelos disponíveis."""
        rag = getattr(request.app.state, "rag", None)
        if rag is None:
            body = ErrorResponse(
                detail=IndexNotReadyError.public_detail,
                error_code=IndexNotReadyError.error_code,
                request_id=get_request_id(),
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        state = rag.readiness()
        content = {
            "status": "ready" if state["ok"] else "not_ready",
            "mode": rag.generator.mode,
            "chunks": rag.chunk_count,
            "checks": state["checks"],
            "request_id": get_request_id(),
        }
        return JSONResponse(status_code=200 if state["ok"] else 503, content=content)

    @app.post("/api/ask", response_model=AskResponse, responses={422: {"model": ErrorResponse}})
    def ask(payload: AskRequest, rag: RAGService = Depends(get_service)) -> AskResponse:
        try:
            result = rag.answer(payload.question)
        except AuroraError:
            raise  # ProviderError → 503, InvalidQuestionError → 422 (handler acima)
        except Exception as exc:
            # Bug inesperado: o stack trace já está no evento provider.error; o cliente recebe 500 genérico.
            raise AuroraError(f"{type(exc).__name__}: {exc}") from exc
        return AskResponse(
            answer=result.answer,
            sources=[SourceResponse(**asdict(source)) for source in result.sources],
            confidence=result.confidence,
            mode=result.mode,
            request_id=result.request_id or get_request_id(),
            timings_ms=result.timings_ms,
            status=result.status,
            refusal_reason=result.refusal_reason,
        )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    return app


app = create_app()
