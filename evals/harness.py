"""Harness de avaliação do pipeline RAG sobre o corpus real.

Carrega ``evals/cases.json``, executa cada caso via ``RAGService.run`` (que expõe o rastro do
retrieval além da resposta) e calcula métricas de recuperação, de resposta/recusa e de latência.

Métricas (definições):

- **recall_at_k** — fração dos casos com fontes esperadas em que algum ``expected_chunk_id``
  (ou, na falta deles, algum documento de ``expected_sources``) aparece entre os ``k`` candidatos
  devolvidos pelo índice (antes dos filtros). Mede o retrieval "cru".
- **selected_recall** — idem, mas sobre os chunks que passaram nos filtros e foram entregues ao
  gerador. Mede o que o LLM efetivamente vê.
- **mrr** — média de 1/posição do primeiro candidato esperado (0 se nenhum aparece em top-k).
- **source_precision** — entre as respostas ``answered`` com fontes esperadas, fração das fontes
  citadas que pertencem a ``expected_sources`` (micro-média sobre todas as fontes citadas).
- **correct_refusal_rate** — em ``out_of_scope``: fração recusada (status ``refused_*``) sem fontes.
- **false_refusal_rate** — em casos com ``expects_sources: true``: fração recusada indevidamente.
- **content_pass_rate** — fração dos casos que satisfazem ``must_contain``/``must_not_contain``
  e ``expects_sources`` (quando não é ``null``).
- **latency_ms** — p50/p95 do tempo total de ``run()`` por caso.

Uso: ``python -m evals.run --mode local`` (ver ``evals/run.py``).
"""

from __future__ import annotations

import dataclasses
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.domain import AnswerStatus, RAGRun
from app.rag import RAGService

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "cases.json"
CATEGORIES = frozenset({"in_scope", "partial", "out_of_scope", "typo", "no_accent", "synonym", "adversarial"})
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    expects_sources: bool | None = None
    notes: str = ""

    @property
    def has_expected_retrieval(self) -> bool:
        return bool(self.expected_chunk_ids or self.expected_sources)


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["cases"] if isinstance(raw, dict) else raw
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for entry in entries:
        case = EvalCase(
            id=str(entry["id"]),
            category=str(entry.get("category", "in_scope")),
            question=str(entry["question"]),
            expected_sources=tuple(entry.get("expected_sources", ())),
            expected_chunk_ids=tuple(entry.get("expected_chunk_ids", ())),
            must_contain=tuple(entry.get("must_contain", ())),
            must_not_contain=tuple(entry.get("must_not_contain", ())),
            expects_sources=entry.get("expects_sources"),
            notes=str(entry.get("notes", "")),
        )
        if case.category not in CATEGORIES:
            raise ValueError(f"caso {case.id}: categoria desconhecida {case.category!r}")
        if case.id in seen:
            raise ValueError(f"id de caso duplicado: {case.id}")
        if case.expects_sources is not None and not isinstance(case.expects_sources, bool):
            raise ValueError(f"caso {case.id}: expects_sources deve ser true, false ou null")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"nenhum caso em {path}")
    return cases


# ---------------------------------------------------------------------------
# Resultado por caso
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _hit(chunk_id: str, case: EvalCase) -> bool:
    if case.expected_chunk_ids:
        return chunk_id in case.expected_chunk_ids
    return chunk_id.split(":", 1)[0] in case.expected_sources


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    status: str
    confidence: str
    answer: str
    cited_sources: list[str]
    candidates: list[dict[str, Any]]
    selected_ids: list[str]
    latency_ms: float
    expects_sources: bool | None = None
    hit_at_k: bool | None = None
    hit_in_selected: bool | None = None
    reciprocal_rank: float | None = None
    source_precision: float | None = None
    content_ok: bool = True
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def refused(self) -> bool:
        return self.status.startswith("refused")


def evaluate_case(service: RAGService, case: EvalCase) -> CaseResult:
    started = time.perf_counter()
    try:
        run: RAGRun = service.run(case.question)
    except Exception as exc:
        return CaseResult(
            id=case.id,
            category=case.category,
            question=case.question,
            status=str(AnswerStatus.ERROR),
            confidence="",
            answer="",
            cited_sources=[],
            candidates=[],
            selected_ids=[],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            expects_sources=case.expects_sources,
            content_ok=False,
            failures=[f"exceção: {type(exc).__name__}: {exc}"],
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    answer = run.answer
    candidates = [{"id": item.chunk.id, "score": round(item.score, 4)} for item in run.retrieval.candidates]
    selected_ids = [item.chunk.id for item in run.retrieval.selected]
    cited = [source.document for source in answer.sources]
    result = CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        status=str(answer.status),
        confidence=str(answer.confidence),
        answer=answer.answer,
        cited_sources=cited,
        candidates=candidates,
        selected_ids=selected_ids,
        latency_ms=latency_ms,
        expects_sources=case.expects_sources,
    )

    if case.has_expected_retrieval:
        candidate_ids = [item.chunk.id for item in run.retrieval.candidates]
        ranks = [index for index, chunk_id in enumerate(candidate_ids, start=1) if _hit(chunk_id, case)]
        result.hit_at_k = bool(ranks)
        result.reciprocal_rank = 1.0 / ranks[0] if ranks else 0.0
        result.hit_in_selected = any(_hit(chunk_id, case) for chunk_id in selected_ids)
        if cited:
            result.source_precision = sum(document in case.expected_sources for document in cited) / len(cited)

    normalized = _norm(answer.answer)
    for fragment in case.must_contain:
        if _norm(fragment) not in normalized:
            result.failures.append(f"must_contain ausente: {fragment!r}")
    for fragment in case.must_not_contain:
        if _norm(fragment) in normalized:
            result.failures.append(f"must_not_contain presente: {fragment!r}")
    if case.expects_sources is True and not answer.sources:
        result.failures.append("esperava fontes, resposta sem fontes")
    if case.expects_sources is False and answer.sources:
        result.failures.append(f"não esperava fontes, citou {cited}")
    if result.refused and answer.sources:
        result.failures.append("recusa com fontes (violação de contrato)")
    result.content_ok = not result.failures
    return result


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarize(results: list[CaseResult], *, k: int) -> dict[str, Any]:
    with_retrieval = [r for r in results if r.hit_at_k is not None]
    answered_with_expectation = [r for r in results if r.source_precision is not None]
    out_of_scope = [r for r in results if r.category == "out_of_scope"]
    must_answer = [r for r in results if r.error is None and r.expects_sources is True]
    latencies = [r.latency_ms for r in results]

    metrics: dict[str, Any] = {
        "cases": len(results),
        "errors": sum(r.error is not None for r in results),
        "k": k,
        "recall_at_k": _mean([float(bool(r.hit_at_k)) for r in with_retrieval]),
        "selected_recall": _mean([float(bool(r.hit_in_selected)) for r in with_retrieval]),
        "mrr": _mean([r.reciprocal_rank or 0.0 for r in with_retrieval]),
        "source_precision": _mean([r.source_precision or 0.0 for r in answered_with_expectation]),
        "correct_refusal_rate": _mean([float(r.refused and not r.cited_sources) for r in out_of_scope]),
        "false_refusal_rate": _mean([float(r.refused) for r in must_answer]),
        "content_pass_rate": _mean([float(r.content_ok) for r in results]),
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "by_category": {},
    }
    for category in sorted({r.category for r in results}):
        subset = [r for r in results if r.category == category]
        subset_retrieval = [r for r in subset if r.hit_at_k is not None]
        metrics["by_category"][category] = {
            "cases": len(subset),
            "recall_at_k": _mean([float(bool(r.hit_at_k)) for r in subset_retrieval]),
            "mrr": _mean([r.reciprocal_rank or 0.0 for r in subset_retrieval]),
            "content_pass_rate": _mean([float(r.content_ok) for r in subset]),
            "refused": sum(r.refused for r in subset),
        }
    return metrics


@dataclass
class EvalReport:
    mode: str
    settings: dict[str, Any]
    metrics: dict[str, Any]
    results: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "settings": self.settings,
            "metrics": self.metrics,
            "results": [asdict(result) for result in self.results],
        }

    def failures(self) -> list[CaseResult]:
        return [result for result in self.results if not result.content_ok]


def run_eval(service: RAGService, cases: list[EvalCase] | None = None) -> EvalReport:
    cases = cases or load_cases()
    results = [evaluate_case(service, case) for case in cases]
    settings = {
        "rag_mode": service.settings.rag_mode,
        "retrieval_k": service.settings.retrieval_k,
        "min_score": service.settings.thresholds.min_score,
        "thresholds": dataclasses.asdict(service.settings.thresholds),
        "embedding_model": service.settings.embedding_model if service.settings.rag_mode == "ollama" else "hash-local",
        "generation_model": (
            service.settings.generation_model if service.settings.rag_mode == "ollama" else "extractive-local"
        ),
        "chunks": service.chunk_count,
    }
    return EvalReport(
        mode=service.generator.mode,
        settings=settings,
        metrics=summarize(results, k=service.settings.retrieval_k),
        results=results,
    )
