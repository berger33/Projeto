"""CLI do harness de avaliação.

Exemplos::

    python -m evals.run                         # modo local (hash embedding + extrativo), corpus corpus/
    python -m evals.run --mode ollama           # modo principal (exige Ollama em OLLAMA_BASE_URL)
    python -m evals.run --mode ollama --k 8 --min-score 0.3
    python -m evals.run --save                  # grava evals/results/<timestamp>-<mode>.json
    python -m evals.run --show-failures         # imprime resposta/candidatos dos casos reprovados

Código de saída: 0 sempre que a execução termina (pisos são aplicados nos testes, não aqui),
exceto quando algum caso lança exceção (exit 1) — útil para detectar Ollama fora do ar.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import ConfigError, Settings
from app.observability import configure_logging
from app.rag import RAGService
from evals.harness import ROOT, EvalReport, load_cases, run_eval

RESULTS_DIR = ROOT / "evals" / "results"


def _fmt(value: float | None, *, pct: bool = True) -> str:
    if value is None:
        return "   n/a"
    return f"{value * 100:5.1f}%" if pct else f"{value:7.1f}"


def print_report(report: EvalReport, *, show_failures: bool) -> None:
    metrics = report.metrics
    print(f"\n=== Avaliação RAG — modo {report.mode} ===")
    print("settings: " + ", ".join(f"{key}={value}" for key, value in report.settings.items()))
    print(f"casos: {metrics['cases']}  erros: {metrics['errors']}  k={metrics['k']}")
    print(f"  recall@k ............ {_fmt(metrics['recall_at_k'])}")
    print(f"  selected recall ..... {_fmt(metrics['selected_recall'])}")
    print(f"  MRR ................. {_fmt(metrics['mrr'])}")
    print(f"  source precision .... {_fmt(metrics['source_precision'])}")
    print(f"  correct refusal ..... {_fmt(metrics['correct_refusal_rate'])}")
    print(f"  false refusal ....... {_fmt(metrics['false_refusal_rate'])}")
    print(f"  content pass ........ {_fmt(metrics['content_pass_rate'])}")
    latency = metrics["latency_ms"]
    print(f"  latency p50/p95 ..... {latency['p50']} / {latency['p95']} ms")
    print("\n  por categoria:")
    print(f"  {'categoria':<14}{'casos':>6}{'recall@k':>10}{'MRR':>8}{'pass':>8}{'recusas':>9}")
    for name, data in metrics["by_category"].items():
        print(
            f"  {name:<14}{data['cases']:>6}{_fmt(data['recall_at_k']):>10}"
            f"{_fmt(data['mrr']):>8}{_fmt(data['content_pass_rate']):>8}{data['refused']:>9}"
        )
    failures = report.failures()
    if failures:
        print(f"\n  reprovados ({len(failures)}):")
        for result in failures:
            print(f"   - {result.id} [{result.category}] status={result.status}: {'; '.join(result.failures)}")
            if show_failures:
                print(f"       Q: {result.question}")
                print(f"       A: {result.answer[:300]}")
                print(f"       candidatos: {result.candidates}")
                print(f"       selecionados: {result.selected_ids}  fontes: {result.cited_sources}")
    print()


def save_report(report: EvalReport, directory: Path | None = None) -> Path:
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{report.settings['rag_mode']}.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia o pipeline RAG sobre evals/cases.json.")
    parser.add_argument("--mode", choices=["local", "ollama"], default=os.getenv("RAG_MODE", "local"))
    parser.add_argument("--docs", default=str(ROOT / "corpus"), help="diretório do corpus (padrão: corpus/)")
    parser.add_argument("--cases", default=None, help="arquivo de casos (padrão: evals/cases.json)")
    parser.add_argument("--k", type=int, default=None, help="sobrescreve RAG_TOP_K")
    parser.add_argument("--min-score", type=float, default=None, help="sobrescreve RAG_MIN_SCORE")
    parser.add_argument("--save", action="store_true", help="grava o relatório em evals/results/")
    parser.add_argument("--show-failures", action="store_true", help="detalha resposta e candidatos dos reprovados")
    parser.add_argument("--json", action="store_true", help="imprime o relatório completo em JSON no stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Silencia os eventos JSON do pipeline por padrão; LOG_LEVEL=INFO/DEBUG reabilita.
    configure_logging(level=os.getenv("LOG_LEVEL", "WARNING"))

    overrides: dict[str, object] = {"rag_mode": args.mode}
    if args.k is not None:
        overrides["retrieval_k"] = args.k
    if args.min_score is not None:
        overrides["min_score"] = args.min_score
    try:
        settings = dataclasses.replace(Settings.from_env(), **overrides)  # type: ignore[arg-type]
    except ConfigError as exc:
        print(f"configuração inválida: {exc}", file=sys.stderr)
        return 2
    try:
        service = RAGService(args.docs, settings)
    except Exception as exc:
        print(f"erro ao construir o índice ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    cases = load_cases(Path(args.cases)) if args.cases else load_cases()
    report = run_eval(service, cases)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_report(report, show_failures=args.show_failures)
    if args.save:
        path = save_report(report)
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"relatório salvo em {shown}")
    return 1 if report.metrics["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
