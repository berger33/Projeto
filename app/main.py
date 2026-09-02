from __future__ import annotations

import logging
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, StringConstraints, field_validator

from .config import Settings
from .errors import AuroraError, IndexNotReadyError
from .observability import RequestContextMiddleware, configure_logging, get_request_id, log_event
from .rag import RAGService

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


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    confidence: str
    mode: str
    request_id: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


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


def create_app(service: RAGService | None = None, *, service_factory: ServiceFactory | None = None) -> FastAPI:
    """Cria a aplicação.

    ``service`` (testes) injeta um serviço já construído. Caso contrário ``service_factory`` (padrão
    ``build_service``) roda **uma única vez** no ``lifespan``, antes de o servidor aceitar conexões.
    Se falhar, o processo não sobe: o erro é logado (``startup.failed``) e propagado para o uvicorn
    encerrar com código de saída ≠ 0 (Fase 2, G-01/G-03).
    """
    configure_logging()
    factory = service_factory or build_service

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
        app.state.rag = None

    app = FastAPI(
        title="Aurora Document RAG API",
        version="2.1.0",
        description="RAG documental com retrieval vetorial, geração local opcional e citações.",
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
    )
    app.state.rag = service  # serviço injetado fica disponível mesmo sem o lifespan (TestClient sem `with`)
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
            sources=[SourceResponse(**source.__dict__) for source in result.sources],
            confidence=result.confidence,
            mode=result.mode,
            request_id=result.request_id or get_request_id(),
            timings_ms=result.timings_ms,
        )

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(
            """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Aurora RAG</title><style>body{font-family:system-ui;background:#f6f4f1;margin:0;padding:40px}.card{max-width:860px;margin:auto;background:#fff;padding:32px;border-radius:20px;box-shadow:0 12px 40px #0001}textarea{width:100%;min-height:100px;padding:12px;box-sizing:border-box}button{padding:12px 18px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}.answer{margin-top:16px;padding:16px;background:#f3f3f3;border-radius:10px;white-space:pre-wrap}</style></head><body><main class='card'><h1>Aurora Document RAG</h1><p>Faça uma pergunta sobre a documentação da loja.</p><textarea id='q'></textarea><p><button onclick='ask()'>Perguntar</button></p><div id='a' class='answer'>Pronto.</div></main><script>async function ask(){const el=document.getElementById('a');el.textContent='Consultando...';const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});const d=await r.json();if(!r.ok){el.textContent=d.detail||'Erro';return}const s=(d.sources||[]).map(x=>x.document+(x.page?' p.'+x.page:'')+(x.row?' linha '+x.row:'')).join(', ');el.textContent=d.answer+(s?'\\n\\nFontes: '+s:'')+'\\nModo: '+d.mode}</script></body></html>"""
        )

    return app


app = create_app()
