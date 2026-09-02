"""Testes do harness de avaliação (evals/) e gate de regressão sobre o corpus real.

Três camadas:

1. Unidade do harness com um serviço fake (métricas calculadas corretamente, casos de erro).
2. Integridade de ``evals/cases.json`` contra o corpus ``docs/`` (ids existem, fragmentos são
   alcançáveis, categorias válidas).
3. Gate de regressão: roda os 53 casos em modo ``local`` sobre ``docs/`` e compara com os pisos
   de ``evals/thresholds.json`` (baseline medida em P0-03). O gate do modo ``ollama`` só roda com
   ``RAG_EVAL_OLLAMA=1`` e servidor acessível.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.documents import load_chunks
from app.domain import Chunk, RAGAnswer, RAGRun, Retrieval, RetrievedChunk, SourceRef
from app.rag import REFUSAL_TEXT, RAGService
from evals import run as eval_cli
from evals.harness import (
    CATEGORIES,
    EvalCase,
    evaluate_case,
    load_cases,
    run_eval,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
THRESHOLDS = json.loads((ROOT / "evals" / "thresholds.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Unidade do harness (serviço fake, sem corpus)
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str = "texto") -> Chunk:
    source, _, locator = chunk_id.partition(":")
    return Chunk(id=chunk_id, source=source, locator=locator, text=text)


class FakeService:
    """Reproduz a interface usada pelo harness: ``run(question) -> RAGRun``."""

    def __init__(self, script: dict[str, RAGRun]) -> None:
        self.script = script
        self.settings = Settings(rag_mode="local", retrieval_k=3, min_score=0.1)
        self.chunk_count = 4
        self.generator = type("G", (), {"mode": "fake"})()

    def run(self, question: str) -> RAGRun:
        outcome = self.script[question]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _run(
    *,
    candidates: list[str],
    selected: list[str] | None = None,
    answer: str = "resposta",
    sources: list[str] | None = None,
    status: str = "answered",
) -> RAGRun:
    retrieved = [RetrievedChunk(chunk=_chunk(cid), score=1.0 - i * 0.1) for i, cid in enumerate(candidates)]
    selected_ids = candidates if selected is None else selected
    chosen = [item for item in retrieved if item.chunk.id in selected_ids]
    refs = [SourceRef(document=doc, row=1) for doc in (sources or [])]
    return RAGRun(
        answer=RAGAnswer(answer=answer, sources=refs, confidence="alta", mode="fake", status=status),
        retrieval=Retrieval(candidates=retrieved, selected=chosen),
    )


@pytest.fixture()
def fake_cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="hit-1",
            category="in_scope",
            question="q1",
            expected_sources=("a.csv",),
            expected_chunk_ids=("a.csv:r1",),
            must_contain=("pix",),
            expects_sources=True,
        ),
        EvalCase(
            id="hit-rank2",
            category="synonym",
            question="q2",
            expected_sources=("a.csv",),
            expected_chunk_ids=("a.csv:r1",),
            expects_sources=True,
        ),
        EvalCase(
            id="miss",
            category="typo",
            question="q3",
            expected_sources=("a.csv",),
            expected_chunk_ids=("a.csv:r1",),
            expects_sources=True,
        ),
        EvalCase(
            id="oos-ok", category="out_of_scope", question="q4", must_contain=("não encontrei",), expects_sources=False
        ),
        EvalCase(
            id="oos-bad", category="out_of_scope", question="q5", must_contain=("não encontrei",), expects_sources=False
        ),
        EvalCase(
            id="false-refusal",
            category="in_scope",
            question="q6",
            expected_sources=("a.csv",),
            must_not_contain=("90 dias",),
            expects_sources=True,
        ),
    ]


@pytest.fixture()
def fake_service() -> FakeService:
    return FakeService(
        {
            # acerto no rank 1, resposta correta, uma fonte certa e uma errada -> precision 0.5
            "q1": _run(candidates=["a.csv:r1", "b.pdf:p1:c1"], answer="Aceitamos PIX", sources=["a.csv", "b.pdf"]),
            # acerto no rank 2 -> RR 0.5; fontes 100 % corretas
            "q2": _run(candidates=["b.pdf:p1:c1", "a.csv:r1"], sources=["a.csv"]),
            # esperado não aparece em top-k -> RR 0, hit False
            "q3": _run(candidates=["b.pdf:p1:c1", "c.pdf:p1:c1"], sources=["b.pdf"]),
            # recusa correta
            "q4": _run(candidates=["b.pdf:p1:c1"], selected=[], answer=REFUSAL_TEXT, status="refused_no_context"),
            # fora de escopo respondido com fonte -> recusa incorreta
            "q5": _run(candidates=["b.pdf:p1:c1"], answer="Canberra", sources=["b.pdf"]),
            # deveria responder mas recusou -> false refusal
            "q6": _run(candidates=["a.csv:r1"], selected=[], answer=REFUSAL_TEXT, status="refused_no_context"),
        }
    )


def test_evaluate_case_computes_retrieval_and_content_checks(fake_service: FakeService, fake_cases: list[EvalCase]):
    hit = evaluate_case(fake_service, fake_cases[0])  # type: ignore[arg-type]
    assert hit.hit_at_k is True
    assert hit.hit_in_selected is True
    assert hit.reciprocal_rank == 1.0
    assert hit.source_precision == 0.5
    assert hit.content_ok is True
    assert hit.latency_ms >= 0

    rank2 = evaluate_case(fake_service, fake_cases[1])  # type: ignore[arg-type]
    assert rank2.reciprocal_rank == 0.5

    miss = evaluate_case(fake_service, fake_cases[2])  # type: ignore[arg-type]
    assert miss.hit_at_k is False
    assert miss.reciprocal_rank == 0.0
    assert miss.source_precision == 0.0

    wrong_oos = evaluate_case(fake_service, fake_cases[4])  # type: ignore[arg-type]
    assert wrong_oos.content_ok is False
    assert any("não esperava fontes" in failure for failure in wrong_oos.failures)
    assert any("must_contain ausente" in failure for failure in wrong_oos.failures)

    false_refusal = evaluate_case(fake_service, fake_cases[5])  # type: ignore[arg-type]
    assert false_refusal.refused is True
    assert any("esperava fontes" in failure for failure in false_refusal.failures)


def test_evaluate_case_normalizes_whitespace_and_case():
    case = EvalCase(id="ws", category="in_scope", question="q", must_contain=("Somente pelo Período",))
    service = FakeService({"q": _run(candidates=[], answer="mantidos   somente\npelo período necessário")})
    result = evaluate_case(service, case)  # type: ignore[arg-type]
    assert result.content_ok is True
    assert result.hit_at_k is None  # caso sem expectativa de retrieval não entra em recall/MRR


def test_evaluate_case_flags_refusal_with_sources_as_contract_violation():
    case = EvalCase(id="contract", category="in_scope", question="q", expected_sources=("a.csv",))
    service = FakeService(
        {"q": _run(candidates=["a.csv:r1"], answer=REFUSAL_TEXT, sources=["a.csv"], status="refused_by_model")}
    )
    result = evaluate_case(service, case)  # type: ignore[arg-type]
    assert any("recusa com fontes" in failure for failure in result.failures)


def test_evaluate_case_records_exception_instead_of_aborting():
    case = EvalCase(id="boom", category="in_scope", question="q", expected_sources=("a.csv",))
    service = FakeService({"q": httpx.ConnectError("connection refused")})  # type: ignore[dict-item]
    result = evaluate_case(service, case)  # type: ignore[arg-type]
    assert result.status == "error"
    assert result.error is not None and "ConnectError" in result.error
    assert result.content_ok is False


def test_summarize_aggregates_expected_values(fake_service: FakeService, fake_cases: list[EvalCase]):
    report = run_eval(fake_service, fake_cases)  # type: ignore[arg-type]
    metrics = report.metrics
    assert metrics["cases"] == 6
    assert metrics["errors"] == 0
    # 4 casos com expectativa de retrieval: hits q1,q2,q6 (3/4); MRR (1 + .5 + 0 + 1)/4
    assert metrics["recall_at_k"] == pytest.approx(0.75)
    assert metrics["mrr"] == pytest.approx(0.625)
    # selected: q1 sim, q2 sim, q3 não, q6 não (selected vazio) -> 2/4
    assert metrics["selected_recall"] == pytest.approx(0.5)
    # precision: q1 0.5, q2 1.0, q3 0.0 -> média 0.5 (q6 recusou sem fontes -> não entra)
    assert metrics["source_precision"] == pytest.approx(0.5)
    assert metrics["correct_refusal_rate"] == pytest.approx(0.5)
    # expects_sources=True: q1,q2,q3,q6 -> 1 recusa em 4
    assert metrics["false_refusal_rate"] == pytest.approx(0.25)
    # content_ok: q1 ok, q2 ok, q3 ok (sem must_*), q4 ok, q5 falha, q6 falha -> 4/6
    assert metrics["content_pass_rate"] == pytest.approx(4 / 6, abs=1e-3)
    assert set(metrics["by_category"]) == {"in_scope", "synonym", "typo", "out_of_scope"}
    assert metrics["by_category"]["out_of_scope"]["refused"] == 1
    assert metrics["latency_ms"]["p50"] is not None and metrics["latency_ms"]["p95"] is not None
    assert report.settings["rag_mode"] == "local"
    assert {result.id for result in report.failures()} == {"oos-bad", "false-refusal"}


def test_summarize_handles_empty_and_single_result():
    assert summarize([], k=5)["recall_at_k"] is None
    single = summarize(
        [evaluate_case(FakeService({"q": _run(candidates=[])}), EvalCase(id="x", category="in_scope", question="q"))],
        k=5,
    )  # type: ignore[arg-type]
    assert single["cases"] == 1
    assert single["latency_ms"]["p50"] == single["latency_ms"]["p95"]


def test_load_cases_rejects_bad_files(tmp_path: Path):
    bad_category = tmp_path / "a.json"
    bad_category.write_text(
        json.dumps({"cases": [{"id": "x", "category": "weird", "question": "q"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="categoria desconhecida"):
        load_cases(bad_category)

    duplicated = tmp_path / "b.json"
    duplicated.write_text(
        json.dumps({"cases": [{"id": "x", "question": "q"}, {"id": "x", "question": "q2"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicado"):
        load_cases(duplicated)

    empty = tmp_path / "c.json"
    empty.write_text(json.dumps({"cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="nenhum caso"):
        load_cases(empty)

    legacy_list = tmp_path / "d.json"
    legacy_list.write_text(json.dumps([{"id": "y", "question": "q", "must_contain": ["a"]}]), encoding="utf-8")
    assert load_cases(legacy_list)[0].category == "in_scope"


# ---------------------------------------------------------------------------
# 2. Integridade de evals/cases.json contra o corpus real
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_cases() -> list[EvalCase]:
    return load_cases()


@pytest.fixture(scope="module")
def real_chunks() -> dict[str, Chunk]:
    return {chunk.id: chunk for chunk in load_chunks(DOCS)}


def test_cases_cover_required_categories(real_cases: list[EvalCase]):
    present = {case.category for case in real_cases}
    assert present == CATEGORIES, f"faltam categorias: {CATEGORIES - present}"
    assert len(real_cases) >= 40
    assert sum(case.category == "out_of_scope" for case in real_cases) >= 5


def test_cases_reference_existing_chunks_and_reachable_fragments(
    real_cases: list[EvalCase], real_chunks: dict[str, Chunk]
):
    documents = {chunk.source for chunk in real_chunks.values()}
    for case in real_cases:
        for document in case.expected_sources:
            assert document in documents, f"{case.id}: documento {document} não existe em docs/"
        for chunk_id in case.expected_chunk_ids:
            assert chunk_id in real_chunks, f"{case.id}: chunk {chunk_id} não existe"
            assert chunk_id.split(":", 1)[0] in case.expected_sources, f"{case.id}: {chunk_id} fora de expected_sources"
        if case.category == "out_of_scope":
            assert not case.expected_sources and case.expects_sources is False, case.id
            continue
        if case.must_contain and case.expected_chunk_ids:
            haystack = " ".join(" ".join(real_chunks[cid].text.split()).lower() for cid in case.expected_chunk_ids)
            for fragment in case.must_contain:
                assert " ".join(fragment.split()).lower() in haystack, (
                    f"{case.id}: must_contain {fragment!r} não ocorre nos chunks esperados"
                )


# ---------------------------------------------------------------------------
# 3. Gate de regressão sobre docs/ (modo local) e gate opcional Ollama
# ---------------------------------------------------------------------------


def _assert_thresholds(metrics: dict[str, Any], thresholds: dict[str, Any]) -> None:
    problems: list[str] = []
    for name, bound in thresholds.items():
        if not isinstance(bound, dict) or ("min" not in bound and "max" not in bound):
            continue  # metadados (measured_baseline, _comment)
        value = metrics["latency_ms"]["p95"] if name == "latency_p95_ms" else metrics[name]
        if value is None:
            problems.append(f"{name}: sem valor")
            continue
        if "min" in bound and value < bound["min"]:
            problems.append(f"{name}={value} < piso {bound['min']}")
        if "max" in bound and value > bound["max"]:
            problems.append(f"{name}={value} > teto {bound['max']}")
    assert not problems, "regressão nas métricas de avaliação:\n  " + "\n  ".join(problems)


@pytest.fixture(scope="module")
def local_report():
    service = RAGService(DOCS, Settings.from_env() if os.getenv("RAG_MODE") == "local" else Settings(rag_mode="local"))
    return run_eval(service)


def test_local_eval_meets_recorded_baseline(local_report):
    assert local_report.metrics["errors"] == 0
    assert local_report.metrics["cases"] >= 40
    _assert_thresholds(local_report.metrics, THRESHOLDS["local"])


def test_local_eval_out_of_scope_never_cites_sources_when_refusing(local_report):
    for result in local_report.results:
        if result.refused:
            assert result.cited_sources == [], f"{result.id}: recusa com fontes"


def test_local_eval_adversarial_cases_do_not_echo_injected_claims(local_report):
    adversarial = [result for result in local_report.results if result.category == "adversarial"]
    assert adversarial
    for result in adversarial:
        assert result.content_ok, f"{result.id}: {result.failures}"


def _ollama_reachable(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.ollama()
def test_ollama_eval_meets_acceptance_targets():
    if os.getenv("RAG_EVAL_OLLAMA") != "1":
        pytest.skip("defina RAG_EVAL_OLLAMA=1 para rodar a avaliação contra o Ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", Settings().ollama_base_url)
    if not _ollama_reachable(base_url):
        pytest.skip(f"Ollama não acessível em {base_url}")
    settings = Settings.from_env()
    if settings.rag_mode != "ollama":
        settings = Settings(
            rag_mode="ollama",
            ollama_base_url=settings.ollama_base_url,
            embedding_model=settings.embedding_model,
            generation_model=settings.generation_model,
            retrieval_k=settings.retrieval_k,
            min_score=settings.min_score,
        )
    report = run_eval(RAGService(DOCS, settings))
    eval_cli.print_report(report, show_failures=True)
    path = eval_cli.save_report(report)
    print(f"relatório salvo em {path}")
    assert report.metrics["errors"] == 0
    _assert_thresholds(report.metrics, THRESHOLDS["ollama"])


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def test_cli_runs_local_mode_and_saves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(eval_cli, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_MODE", raising=False)
    exit_code = eval_cli.main(["--mode", "local", "--save", "--show-failures"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Avaliação RAG — modo local-extractive" in captured.out
    assert "por categoria" in captured.out
    saved = list(tmp_path.glob("*-local.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["metrics"]["cases"] >= 40
    assert payload["results"][0]["candidates"]


def test_cli_json_output_and_overrides(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv("RAG_MODE", raising=False)
    assert eval_cli.main(["--mode", "local", "--json", "--k", "3", "--min-score", "0.5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["retrieval_k"] == 3
    assert payload["settings"]["min_score"] == 0.5
    assert payload["metrics"]["k"] == 3


def test_cli_reports_index_build_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert eval_cli.main(["--mode", "local", "--docs", str(tmp_path / "missing")]) == 1
    assert "erro ao construir o índice" in capsys.readouterr().err


def test_results_directory_is_ignored_except_gitkeep():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "evals/results/*.json" in gitignore
    assert "!evals/results/.gitkeep" in gitignore
    assert (ROOT / "evals" / "results" / ".gitkeep").exists()
