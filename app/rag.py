from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Settings
from .documents import load_corpus
from .domain import (
    REFUSAL_TEXT,
    AnswerStatus,
    Confidence,
    Generation,
    RAGAnswer,
    RAGRun,
    Retrieval,
    RetrievedChunk,
)
from .embeddings import EmbeddingProvider, HashEmbeddingProvider, OllamaEmbeddingProvider
from .errors import InvalidQuestionError, ProviderError, ping_ollama
from .generation import AnswerGenerator, ExtractiveGenerator, OllamaGenerator, PromptBudget
from .observability import Timings, get_request_id, log_event, request_context
from .refusal import judge
from .retrieval import VectorIndex
from .retriever import HybridRetriever, RetrieverConfig
from .sources import derive_sources, strip_citations

__all__ = ["REFUSAL_TEXT", "RAGService"]

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, docs_dir: str | Path, settings: Settings):
        self.settings = settings
        self.docs_dir = Path(docs_dir)
        started = time.perf_counter()
        try:
            chunks, ingest = load_corpus(docs_dir)
            self.ingest_report = ingest
            embeddings: EmbeddingProvider
            generator: AnswerGenerator
            if settings.rag_mode == "ollama":
                embeddings = OllamaEmbeddingProvider(
                    settings.ollama_base_url,
                    settings.embedding_model,
                    timeout=settings.embed_timeout_s,
                    batch_size=settings.embed_batch_size,
                )
                generator = OllamaGenerator(
                    settings.ollama_base_url,
                    settings.generation_model,
                    timeout=settings.generate_timeout_s,
                    budget=PromptBudget(num_ctx=settings.num_ctx, num_predict=settings.num_predict),
                )
            else:
                embeddings = HashEmbeddingProvider()
                generator = ExtractiveGenerator()
            self.generator = generator
            self.index = VectorIndex(chunks, embeddings)
            thresholds = settings.thresholds
            self.thresholds = thresholds
            self.retriever = HybridRetriever(
                chunks,
                self.index,
                RetrieverConfig(
                    k=settings.retrieval_k,
                    min_score=thresholds.min_score,
                    vector_only_min_score=thresholds.vector_only_min_score,
                    vector_with_overlap_min_score=thresholds.vector_with_overlap_min_score,
                    min_lexical_coverage=thresholds.min_lexical_coverage,
                ),
            )
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
            duplicates_removed=len(ingest.duplicates),
            skipped_files=ingest.skipped or None,
            dimension=self.index.dimension,
            query_prefix=embeddings.prefixes.query if isinstance(embeddings, OllamaEmbeddingProvider) else None,
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
            fused = self.retriever.fuse(question)
        with timings.stage("filter"):
            selected = [
                RetrievedChunk(chunk=item.chunk, score=item.vector_score)
                for item in fused
                if self.retriever.accepts(item)
            ][: self.settings.retrieval_k]
        # ``candidates`` expõe o pool fundido (ordem RRF) com o cosseno como score, para diagnóstico/eval.
        candidates = [RetrievedChunk(chunk=item.chunk, score=item.vector_score) for item in fused]
        log_event(
            logger,
            logging.INFO,
            "query.retrieved",
            question_chars=len(question),
            k=self.settings.retrieval_k,
            min_score=self.thresholds.min_score,
            pool=len(fused),
            candidates=[
                {
                    "id": item.chunk.id,
                    "score": round(item.vector_score, 4),
                    "vector_rank": item.vector_rank,
                    "lexical_rank": item.lexical_rank,
                    "overlap": item.lexical_overlap,
                    "coverage": item.lexical_coverage,
                    "rrf": round(item.fused_score, 5),
                }
                for item in fused[: self.settings.retrieval_k * 2]
            ],
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
                result = self._refusal(AnswerStatus.REFUSED_NO_CONTEXT, request_id, timings, reason="no_context")
                self._log_answer(question, result, started=started)
                return RAGRun(answer=result, retrieval=retrieval)
            with timings.stage("generate"):
                generation = self.generator.generate(question, selected)
            with timings.stage("verify"):
                verdict = judge(generation, selected)
            if verdict.refused:
                # Recusa sempre sai com o texto canônico e sem fontes; o que o modelo escreveu fica no log.
                result = self._refusal(
                    AnswerStatus.REFUSED_BY_MODEL, request_id, timings, reason=verdict.reason, support=verdict.support
                )
                self._log_answer(question, result, started=started, generation=generation)
                log_event(
                    logger,
                    logging.INFO,
                    "answer.refused",
                    reason=verdict.reason,
                    support=verdict.support,
                    structured=generation.structured,
                    declared_grounded=generation.grounded,
                    matched_pattern=verdict.matched_pattern,
                    unsupported_numbers=list(verdict.numbers),
                    model_text_chars=len(generation.text),
                )
                return RAGRun(answer=result, retrieval=retrieval)
            sources, sources_reason = derive_sources(generation, selected)
            result = RAGAnswer(
                answer=strip_citations(generation.text),
                sources=sources,
                confidence=self._confidence(selected, support=verdict.support),
                mode=self.generator.mode,
                request_id=request_id,
                timings_ms=timings.as_dict(),
                status=AnswerStatus.ANSWERED,
                support=verdict.support,
            )
            self._log_answer(question, result, started=started, generation=generation, sources_reason=sources_reason)
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

    def _confidence(self, selected: list[RetrievedChunk], *, support: float | None) -> Confidence:
        """Confiança a partir de três sinais (R-25): score do top-1 na escala do provider, destaque do
        top-1 sobre o top-2 (gap relativo) ou concordância de fontes, e sustentação medida da resposta.

        - **alta**: top-1 >= high_confidence_score **e** (gap relativo >= relative_gap **ou** >= 2 chunks
          selecionados do mesmo documento) **e** sustentação >= 0,8 (quando medida).
        - **baixa**: sustentação medida < 0,6 ou top-1 abaixo do piso vetorial (aceito só por evidência lexical).
        - **média**: o restante.
        """
        thresholds = self.thresholds
        top = selected[0].score
        second = selected[1].score if len(selected) > 1 else 0.0
        gap = (top - second) / top if top > 0 else 0.0
        documents = [item.chunk.source for item in selected]
        agreeing = any(documents.count(document) >= 2 for document in set(documents))
        if support is not None and support < 0.6:
            return Confidence.BAIXA
        if top < thresholds.min_score:
            return Confidence.BAIXA
        strong_top = top >= thresholds.high_confidence_score
        distinct = gap >= thresholds.relative_gap or agreeing
        well_supported = support is None or support >= 0.8
        if strong_top and distinct and well_supported:
            return Confidence.ALTA
        return Confidence.MEDIA

    def _refusal(
        self,
        status: AnswerStatus,
        request_id: str | None,
        timings: Timings,
        *,
        reason: str | None,
        support: float | None = None,
    ) -> RAGAnswer:
        return RAGAnswer(
            answer=REFUSAL_TEXT,
            sources=[],
            confidence=Confidence.BAIXA,
            mode=self.generator.mode,
            request_id=request_id,
            timings_ms=timings.as_dict(),
            status=status,
            refusal_reason=reason,
            support=support,
        )

    def _log_answer(
        self,
        question: str,
        result: RAGAnswer,
        *,
        started: float,
        generation: Generation | None = None,
        sources_reason: str | None = None,
    ) -> None:
        log_event(
            logger,
            logging.INFO,
            "query.answered",
            status=str(result.status),
            mode=result.mode,
            confidence=str(result.confidence),
            sources=len(result.sources),
            source_ids=[source.chunk_id for source in result.sources],
            sources_reason=sources_reason,
            answer_chars=len(result.answer),
            refusal_reason=result.refusal_reason,
            support=result.support,
            timings_ms=result.timings_ms,
            total_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        # Texto integral só em DEBUG: perguntas podem conter dados pessoais.
        if logger.isEnabledFor(logging.DEBUG):
            log_event(
                logger,
                logging.DEBUG,
                "query.text",
                question=question,
                answer=result.answer,
                model_text=generation.text if generation is not None else None,
            )


def _model_available(model: str, available: list[str]) -> bool:
    """``qwen3:1.7b`` casa com ``qwen3:1.7b``; ``nomic-embed-text`` casa com ``nomic-embed-text:latest``."""
    wanted = model if ":" in model else f"{model}:latest"
    return any(name in (wanted, model) for name in available)
