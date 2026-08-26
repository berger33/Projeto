from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .rag import RAGService

BASE = Path(__file__).resolve().parents[1]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class SourceResponse(BaseModel):
    document: str
    page: int | None = None
    row: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    confidence: str
    mode: str


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService(BASE / "docs", Settings.from_env())


def create_app(service: RAGService | None = None) -> FastAPI:
    app = FastAPI(
        title="Aurora Document RAG API",
        version="2.0.0",
        description="RAG documental com retrieval vetorial, geração local opcional e citações.",
    )

    def dependency() -> RAGService:
        return service or get_service()

    @app.get("/health")
    def health(rag: RAGService = Depends(dependency)):
        return {"status": "ok", "chunks": rag.chunk_count, "mode": rag.generator.mode}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest, rag: RAGService = Depends(dependency)):
        try:
            result = rag.answer(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG indisponível: {exc}") from exc
        return AskResponse(
            answer=result.answer,
            sources=[SourceResponse(**source.__dict__) for source in result.sources],
            confidence=result.confidence,
            mode=result.mode,
        )

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(
            """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Aurora RAG</title><style>body{font-family:system-ui;background:#f6f4f1;margin:0;padding:40px}.card{max-width:860px;margin:auto;background:#fff;padding:32px;border-radius:20px;box-shadow:0 12px 40px #0001}textarea{width:100%;min-height:100px;padding:12px;box-sizing:border-box}button{padding:12px 18px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}.answer{margin-top:16px;padding:16px;background:#f3f3f3;border-radius:10px;white-space:pre-wrap}</style></head><body><main class='card'><h1>Aurora Document RAG</h1><p>Faça uma pergunta sobre a documentação da loja.</p><textarea id='q'></textarea><p><button onclick='ask()'>Perguntar</button></p><div id='a' class='answer'>Pronto.</div></main><script>async function ask(){const el=document.getElementById('a');el.textContent='Consultando...';const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});const d=await r.json();if(!r.ok){el.textContent=d.detail||'Erro';return}const s=(d.sources||[]).map(x=>x.document+(x.page?' p.'+x.page:'')+(x.row?' linha '+x.row:'')).join(', ');el.textContent=d.answer+(s?'\\n\\nFontes: '+s:'')+'\\nModo: '+d.mode}</script></body></html>"""
        )

    return app


app = create_app()
