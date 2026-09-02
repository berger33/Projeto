"""Calibração de limiares de retrieval/confiança a partir do eval (P1-06, R-11).

Varre combinações de limiares sobre ``evals/cases.json`` e reporta, para cada uma, a curva
recusa-correta x recusa-indevida junto com recall/precisão, para escolher o ponto de operação de um
provider. O retrieval (embeddings + BM25) é executado **uma vez** por pergunta; só os filtros e a
confiança são recalculados por combinação, então a varredura é barata mesmo em modo Ollama.

Uso::

    python -m evals.calibrate                       # modo local, grade padrão
    python -m evals.calibrate --mode ollama         # exige Ollama acessível
    python -m evals.calibrate --min-score 0.1,0.12,0.15 --coverage 0.15,0.2,0.25
    python -m evals.calibrate --save                # grava evals/results/<timestamp>-calibration-<modo>.json

A escolha é reportada segundo uma ordem lexicográfica explícita: (1) recusa correta máxima,
(2) recusa indevida mínima, (3) selected recall máximo, (4) precisão de fontes máxima. Os valores
escolhidos devem ser copiados para ``THRESHOLD_PROFILES`` em ``app/config.py`` e para a seção
``profiles`` de ``evals/thresholds.json`` (o teste ``test_profiles_match_config`` garante a coerência).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import THRESHOLD_PROFILES, ConfigError, RetrievalThresholds, Settings
from app.observability import configure_logging
from app.rag import RAGService
from app.retriever import RetrieverConfig, ScoredCandidate
from evals.harness import ROOT, EvalCase, load_cases

RESULTS_DIR = ROOT / "evals" / "results"

DEFAULT_GRID: dict[str, list[float]] = {
    "min_score": [0.08, 0.12, 0.2, 0.3, 0.35, 0.45],
    "vector_only_min_score": [0.45, 0.5, 0.6, 0.65, 0.75],
    "vector_with_overlap_min_score": [0.3, 0.35, 0.45, 0.5, 0.6],
    "min_lexical_coverage": [0.15, 0.2, 0.25, 0.3],
}


@dataclass(frozen=True)
class Outcome:
    correct_refusal_rate: float | None
    false_refusal_rate: float | None
    selected_recall: float | None
    source_precision: float | None
    answered: int
    refused: int

    def key(self) -> tuple[float, float, float, float]:
        return (
            self.correct_refusal_rate or 0.0,
            -(self.false_refusal_rate or 0.0),
            self.selected_recall or 0.0,
            self.source_precision or 0.0,
        )


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _hit(candidate: ScoredCandidate, case: EvalCase) -> bool:
    if case.expected_chunk_ids:
        return candidate.chunk.id in case.expected_chunk_ids
    return candidate.chunk.source in case.expected_sources


def evaluate_thresholds(
    fused: dict[str, list[ScoredCandidate]], cases: list[EvalCase], thresholds: RetrievalThresholds, *, k: int
) -> Outcome:
    """Aplica os filtros de ``RetrieverConfig`` (mesma lógica de ``HybridRetriever.accepts``) ao pool."""
    config = RetrieverConfig(
        k=k,
        min_score=thresholds.min_score,
        vector_only_min_score=thresholds.vector_only_min_score,
        vector_with_overlap_min_score=thresholds.vector_with_overlap_min_score,
        min_lexical_coverage=thresholds.min_lexical_coverage,
    )

    def accepts(candidate: ScoredCandidate) -> bool:
        if candidate.vector_score < config.min_score:
            return False
        if candidate.lexical_coverage >= config.min_lexical_coverage:
            return True
        if candidate.lexical_overlap >= 1 and candidate.vector_score >= config.vector_with_overlap_min_score:
            return True
        return candidate.vector_score >= config.vector_only_min_score

    out_of_scope_refused: list[float] = []
    must_answer_refused: list[float] = []
    selected_hits: list[float] = []
    precisions: list[float] = []
    answered = refused = 0
    for case in cases:
        selected = [candidate for candidate in fused[case.id] if accepts(candidate)][:k]
        is_refused = not selected
        answered += not is_refused
        refused += is_refused
        if case.category == "out_of_scope":
            out_of_scope_refused.append(float(is_refused))
        if case.expects_sources is True:
            must_answer_refused.append(float(is_refused))
        if case.has_expected_retrieval:
            selected_hits.append(float(any(_hit(candidate, case) for candidate in selected)))
            if selected:
                cited = [candidate.chunk.source for candidate in selected[:3]]
                precisions.append(sum(document in case.expected_sources for document in cited) / len(cited))
    return Outcome(
        correct_refusal_rate=_mean(out_of_scope_refused),
        false_refusal_rate=_mean(must_answer_refused),
        selected_recall=_mean(selected_hits),
        source_precision=_mean(precisions),
        answered=answered,
        refused=refused,
    )


def sweep(
    service: RAGService, cases: list[EvalCase], grid: dict[str, list[float]]
) -> tuple[list[tuple[RetrievalThresholds, Outcome]], dict[str, list[ScoredCandidate]]]:
    fused = {case.id: service.retriever.fuse(case.question) for case in cases}
    base = service.settings.thresholds
    results: list[tuple[RetrievalThresholds, Outcome]] = []
    names = list(grid)
    for values in itertools.product(*(grid[name] for name in names)):
        candidate = dataclasses.replace(base, **dict(zip(names, values, strict=True)))
        results.append((candidate, evaluate_thresholds(fused, cases, candidate, k=service.settings.retrieval_k)))
    results.sort(key=lambda pair: pair[1].key(), reverse=True)
    return results, fused


def score_distribution(fused: dict[str, list[ScoredCandidate]], cases: list[EvalCase]) -> dict[str, Any]:
    """Cosseno do top-1 para perguntas legítimas vs. fora de escopo — mostra a escala do provider."""
    legit = [fused[case.id][0].vector_score for case in cases if case.expects_sources is True and fused[case.id]]
    oos = [fused[case.id][0].vector_score for case in cases if case.category == "out_of_scope" and fused[case.id]]

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        return {
            "min": round(min(values), 4),
            "median": round(statistics.median(values), 4),
            "max": round(max(values), 4),
        }

    return {"top1_in_scope": summary(legit), "top1_out_of_scope": summary(oos)}


def _parse_grid(args: argparse.Namespace) -> dict[str, list[float]]:
    grid = dict(DEFAULT_GRID)
    for name in grid:
        raw = getattr(args, name.replace("min_lexical_coverage", "coverage"), None)
        if raw:
            grid[name] = [float(value) for value in raw.split(",")]
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibra limiares de retrieval a partir de evals/cases.json.")
    parser.add_argument("--mode", choices=["local", "ollama"], default=os.getenv("RAG_MODE", "local"))
    parser.add_argument("--docs", default=str(ROOT / "corpus"))
    parser.add_argument("--cases", default=None)
    parser.add_argument("--min-score", dest="min_score", default=None, help="lista separada por vírgula")
    parser.add_argument("--vector-only", dest="vector_only_min_score", default=None)
    parser.add_argument("--vector-with-overlap", dest="vector_with_overlap_min_score", default=None)
    parser.add_argument("--coverage", dest="coverage", default=None)
    parser.add_argument("--top", type=int, default=10, help="quantas combinações mostrar")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=os.getenv("LOG_LEVEL", "WARNING"))
    try:
        settings = dataclasses.replace(Settings.from_env(), rag_mode=args.mode)
    except ConfigError as exc:
        print(f"configuração inválida: {exc}", file=sys.stderr)
        return 2
    try:
        service = RAGService(args.docs, settings)
    except Exception as exc:
        print(f"erro ao construir o índice ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    cases = load_cases(Path(args.cases)) if args.cases else load_cases()
    grid = _parse_grid(args)
    results, fused = sweep(service, cases, grid)
    distribution = score_distribution(fused, cases)
    current = THRESHOLD_PROFILES[args.mode]
    current_outcome = evaluate_thresholds(fused, cases, current, k=settings.retrieval_k)

    report = {
        "mode": args.mode,
        "embedding_model": settings.public_dict()["embedding_model"],
        "cases": len(cases),
        "grid": grid,
        "score_distribution": distribution,
        "current_profile": {"thresholds": dataclasses.asdict(current), "outcome": dataclasses.asdict(current_outcome)},
        "best": [
            {"thresholds": dataclasses.asdict(thresholds), "outcome": dataclasses.asdict(outcome)}
            for thresholds, outcome in results[: args.top]
        ],
        "combinations": len(results),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print(report)
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_DIR / f"{stamp}-calibration-{args.mode}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"relatório salvo em {shown}")
    return 0


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}"


def _print(report: dict[str, Any]) -> None:
    print(f"\n=== Calibração de limiares — modo {report['mode']} ({report['embedding_model']}) ===")
    dist = report["score_distribution"]
    print(f"cosseno top-1  in_scope: {dist['top1_in_scope']}   out_of_scope: {dist['top1_out_of_scope']}")
    print(f"combinações avaliadas: {report['combinations']}  casos: {report['cases']}\n")
    header = f"{'min':>5} {'v_only':>6} {'v_ovl':>6} {'cov':>5} | {'recusa ok':>9} {'recusa ind':>10} {'sel.recall':>10} {'precisão':>8} {'resp/rec':>9}"
    print(header)
    print("-" * len(header))

    def row(entry: dict[str, Any], mark: str = "") -> str:
        t, o = entry["thresholds"], entry["outcome"]
        return (
            f"{t['min_score']:>5.2f} {t['vector_only_min_score']:>6.2f} {t['vector_with_overlap_min_score']:>6.2f} "
            f"{t['min_lexical_coverage']:>5.2f} | {_fmt(o['correct_refusal_rate']):>9} {_fmt(o['false_refusal_rate']):>10} "
            f"{_fmt(o['selected_recall']):>10} {_fmt(o['source_precision']):>8} {o['answered']:>4}/{o['refused']:<4}{mark}"
        )

    print(row(report["current_profile"], "  <- perfil atual"))
    print("-" * len(header))
    for entry in report["best"]:
        print(row(entry))
    print()


if __name__ == "__main__":
    sys.exit(main())
