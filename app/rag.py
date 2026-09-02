from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from .config import Settings
from .documents import load_chunks
from .domain import RAGAnswer, RAGRun, Retrieval, RetrievedChunk, SourceRef
from .embeddings import EmbeddingProvider, HashEmbeddingProvider, OllamaEmbeddingProvider
from .errors import InvalidQuestionError, ProviderError, ping_ollama
from .generation import AnswerGenerator, ExtractiveGenerator, OllamaGenerator
from .observability import Timings, get_request_id, log_event, request_context
from .retrieval import VectorIndex

logger = logging.getLogger(__name__)

STOPWORDS = {
    "qual",
    "quais",
    "como",
    "para",
    "com",
    "uma",
    "uns",
    "das",
    "dos",
    "que",
    "por",
    "ser",
    "sao",
    "são",
    "esta",
    "está",
    "meu",
    "minha",
}


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


REFUSAL_TEXT = "Não encontrei informação suficiente na documentação oficial da Aurora Moda Online."


class RAGService:
    def __init__(self, docs_dir: str | Path, settings: Settings):
        self.settings = settings
        self.docs_dir = Path(docs_dir)
        started = time.perf_counter()
        try:
            chunks = load_chunks(docs_dir)
            embeddings: EmbeddingProvider
            generator: AnswerGenerator
            if settings.rag_mode == "ollama":
                embeddings = OllamaEmbeddingProvider(
                    settings.ollama_base_url, settings.embedding_model, timeout=settings.embed_timeout_s
                )
                generator = OllamaGenerator(
                    settings.ollama_base_url, settings.generation_model, timeout=settings.generate_timeout_s
                )
            else:
                embeddings = HashEmbeddingProvider()
                generator = ExtractiveGenerator()
            self.generator = generator
            self.index = VectorIndex(chunks, embeddings)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "index.error",
                exc_info=exc,
                docs_dir=str(docs_dir),
                mode=settings.rag_mode,
                error_type=type(exc).__name__,
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "index.built",
            docs_dir=str(docs_dir),
            mode=self.generator.mode,
            documents=len({chunk.source for chunk in chunks}),
            chunks=len(chunks),
            dimension=len(self.index.vectors[0]) if self.index.vectors else 0,
            embedding_model=settings.embedding_model if settings.rag_mode == "ollama" else "hash-local",
            generation_model=settings.generation_model if settings.rag_mode == "ollama" else "extractive-local",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    @property
    def chunk_count(self) -> int:
        return len(self.index.chunks)

    def readiness(self) -> dict[str, object]:
        """Estado das dependências para ``/ready``: índice com chunks e, em modo Ollama, servidor
        acessível com os modelos configurados presentes. Nunca levanta; ``ok`` resume o resultado."""
        checks: dict[str, object] = {"index": {"ok": self.chunk_count > 0, "chunks": self.chunk_count}}
        if self.settings.rag_mode == "ollama":
            check: dict[str, object]
            try:
                available = ping_ollama(self.settings.ollama_base_url)
            except ProviderError as exc:
                log_event(
                    logger, logging.WARNING, "ready.ollama_unreachable", error_code=exc.error_code, error=str(exc)
                )
                check = {"ok": False, "error_code": exc.error_code}
            else:
                missing = [
                    model
                    for model in (self.settings.embedding_model, self.settings.generation_model)
                    if not _model_available(model, available)
                ]
                check = {"ok": not missing, "missing_models": missing}
            checks["ollama"] = check
        ok = all(bool(item["ok"]) for item in checks.values() if isinstance(item, dict))
        return {"ok": ok, "checks": checks}

    def answer(self, question: str) -> RAGAnswer:
        return self.run(question).answer

    def run(self, question: str) -> RAGRun:
        """Executa o pipeline completo e devolve a resposta junto com o rastro do retrieval."""
        # Fora de uma requisição HTTP (CLI, evals, testes) cada execução ganha o próprio request_id.
        if get_request_id() is not None:
            return self._run(question)
        with request_context():
            return self._run(question)

    def _retrieve(self, question: str, timings: Timings) -> Retrieval:
        with timings.stage("retrieve"):
            candidates = self.index.search(question, k=self.settings.retrieval_k)
        with timings.stage("filter"):
            question_terms = _terms(question)
            selected = [
                item
                for item in candidates
                if item.score >= self.settings.min_score and question_terms & _terms(item.chunk.text)
            ]
        log_event(
            logger,
            logging.INFO,
            "query.retrieved",
            question_chars=len(question),
            k=self.settings.retrieval_k,
            min_score=self.settings.min_score,
            candidates=[{"id": item.chunk.id, "score": round(item.score, 4)} for item in candidates],
            selected=[item.chunk.id for item in selected],
        )
        return Retrieval(candidates=candidates, selected=selected)

    def _run(self, question: str) -> RAGRun:
        question = question.strip()
        if not question:
            raise InvalidQuestionError("A pergunta não pode ser vazia.")
        timings = Timings()
        request_id = get_request_id()
        started = time.perf_counter()
        try:
            retrieval = self._retrieve(question, timings)
            selected = retrieval.selected
            if not selected:
                result = RAGAnswer(
                    answer=REFUSAL_TEXT,
                    sources=[],
                    confidence="baixa",
                    mode=self.generator.mode,
                    request_id=request_id,
                    timings_ms=timings.as_dict(),
                    status="refused_no_context",
                )
                self._log_answer(question, result, started=started)
                return RAGRun(answer=result, retrieval=retrieval)
            with timings.stage("generate"):
                answer = self.generator.generate(question, selected)
            if answer.lower().startswith("não encontrei informação suficiente"):
                result = RAGAnswer(
                    answer=answer,
                    sources=[],
                    confidence="baixa",
                    mode=self.generator.mode,
                    request_id=request_id,
                    timings_ms=timings.as_dict(),
                    status="refused_by_model",
                )
                self._log_answer(question, result, started=started)
                return RAGRun(answer=result, retrieval=retrieval)
            result = RAGAnswer(
                answer=answer,
                sources=self._sources(selected),
                confidence="alta" if selected[0].score >= 0.45 else "média",
                mode=self.generator.mode,
                request_id=request_id,
                timings_ms=timings.as_dict(),
                status="answered",
            )
            self._log_answer(question, result, started=started)
            return RAGRun(answer=result, retrieval=retrieval)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "provider.error",
                exc_info=exc,
                stage=next(reversed(timings.as_dict()), "retrieve"),
                mode=self.generator.mode,
                error_type=type(exc).__name__,
                timings_ms=timings.as_dict(),
            )
            raise

    @staticmethod
    def _sources(selected: list[RetrievedChunk]) -> list[SourceRef]:
        sources: list[SourceRef] = []
        for item in selected[:3]:
            ref = SourceRef(
                document=item.chunk.source, page=item.chunk.locator.get("page"), row=item.chunk.locator.get("row")
            )
            if ref not in sources:
                sources.append(ref)
        return sources

    def _log_answer(self, question: str, result: RAGAnswer, *, started: float) -> None:
        log_event(
            logger,
            logging.INFO,
            "query.answered",
            status=result.status,
            mode=result.mode,
            confidence=result.confidence,
            sources=len(result.sources),
            answer_chars=len(result.answer),
            timings_ms=result.timings_ms,
            total_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        # Texto integral só em DEBUG: perguntas podem conter dados pessoais.
        if logger.isEnabledFor(logging.DEBUG):
            log_event(logger, logging.DEBUG, "query.text", question=question, answer=result.answer)


def _model_available(model: str, available: list[str]) -> bool:
    """``qwen3:1.7b`` casa com ``qwen3:1.7b``; ``nomic-embed-text`` casa com ``nomic-embed-text:latest``."""
    wanted = model if ":" in model else f"{model}:latest"
    return any(name in (wanted, model) for name in available)
