"""P1-06: limiares por provider calibrados pelo eval; confiança com gap top-1/top-2 e concordância.

Findings: R-11 (0,12/0,45 calibrados para o hash e aplicados a qualquer provider), R-25 (confidence
com 3 valores e 2 causas).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from tests.conftest import ListHandler

from app.config import THRESHOLD_PROFILES, ConfigError, RetrievalThresholds, Settings
from app.domain import Chunk, Confidence, RetrievedChunk
from app.rag import RAGService
from evals import calibrate
from evals.harness import EvalCase, load_cases

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_FILE = json.loads((ROOT / "evals" / "thresholds.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Perfis por provider
# ---------------------------------------------------------------------------


def test_profiles_exist_for_each_mode_and_match_thresholds_json() -> None:
    assert set(THRESHOLD_PROFILES) == {"local", "ollama"}
    for mode, profile in THRESHOLD_PROFILES.items():
        assert dataclasses.asdict(profile) == THRESHOLDS_FILE["profiles"][mode], mode
    # A escala do provider denso é mais alta que a do hash em todos os pisos vetoriais.
    local, ollama = THRESHOLD_PROFILES["local"], THRESHOLD_PROFILES["ollama"]
    assert ollama.min_score > local.min_score
    assert ollama.vector_only_min_score > local.vector_only_min_score
    assert ollama.high_confidence_score > local.high_confidence_score


def test_settings_resolve_profile_by_mode_with_env_overrides() -> None:
    assert Settings(rag_mode="local").thresholds == THRESHOLD_PROFILES["local"]
    assert Settings(rag_mode="ollama").thresholds == THRESHOLD_PROFILES["ollama"]
    custom = Settings.from_env({"RAG_MODE": "ollama", "RAG_MIN_SCORE": "0.4", "RAG_HIGH_CONFIDENCE_SCORE": "0.8"})
    assert custom.thresholds == dataclasses.replace(
        THRESHOLD_PROFILES["ollama"], min_score=0.4, high_confidence_score=0.8
    )
    assert custom.public_dict()["thresholds"]["min_score"] == 0.4
    assert Settings.from_env({"RAG_MIN_SCORE": "  "}).min_score is None  # vazio = perfil


@pytest.mark.parametrize(
    "variable",
    [
        "RAG_MIN_SCORE",
        "RAG_VECTOR_ONLY_MIN_SCORE",
        "RAG_VECTOR_WITH_OVERLAP_MIN_SCORE",
        "RAG_MIN_LEXICAL_COVERAGE",
        "RAG_HIGH_CONFIDENCE_SCORE",
        "RAG_RELATIVE_GAP",
    ],
)
def test_threshold_overrides_are_validated(variable: str) -> None:
    with pytest.raises(ConfigError, match=variable):
        Settings.from_env({variable: "1.5"})
    with pytest.raises(ConfigError, match=variable):
        Settings.from_env({variable: "abc"})


def test_service_uses_profile_thresholds_in_retriever() -> None:
    service = RAGService(ROOT / "corpus", Settings())
    config = service.retriever.config
    profile = THRESHOLD_PROFILES["local"]
    assert (
        config.min_score,
        config.vector_only_min_score,
        config.vector_with_overlap_min_score,
        config.min_lexical_coverage,
    ) == (
        profile.min_score,
        profile.vector_only_min_score,
        profile.vector_with_overlap_min_score,
        profile.min_lexical_coverage,
    )


# ---------------------------------------------------------------------------
# Confiança (R-25)
# ---------------------------------------------------------------------------


def _selected(*pairs: tuple[str, float]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(Chunk(id=f"{doc}:r{i}", text="t", source=doc, locator={}), score)
        for i, (doc, score) in enumerate(pairs)
    ]


@pytest.fixture()
def service() -> RAGService:
    return RAGService(ROOT / "corpus", Settings())


def test_confidence_high_requires_strong_top_and_distinct_or_agreeing_sources(service: RAGService) -> None:
    high = service.thresholds.high_confidence_score
    # top forte + gap grande
    assert service._confidence(_selected(("a.pdf", high + 0.1), ("b.pdf", 0.2)), support=1.0) is Confidence.ALTA
    # top forte, gap pequeno, mas dois chunks do mesmo documento concordam
    assert (
        service._confidence(_selected(("a.pdf", high + 0.05), ("a.pdf", high + 0.04)), support=1.0) is Confidence.ALTA
    )
    # top forte, gap pequeno, documentos diferentes → média
    assert (
        service._confidence(_selected(("a.pdf", high + 0.05), ("b.pdf", high + 0.04)), support=1.0) is Confidence.MEDIA
    )
    # top fraco mesmo com gap
    assert service._confidence(_selected(("a.pdf", high - 0.05), ("b.pdf", 0.1)), support=1.0) is Confidence.MEDIA
    # único chunk forte (gap = 1.0)
    assert service._confidence(_selected(("a.pdf", high + 0.2)), support=None) is Confidence.ALTA


def test_confidence_low_when_support_is_weak_or_top_below_vector_floor(service: RAGService) -> None:
    high = service.thresholds.high_confidence_score
    assert service._confidence(_selected(("a.pdf", high + 0.2), ("b.pdf", 0.1)), support=0.5) is Confidence.BAIXA
    assert service._confidence(_selected(("a.pdf", high + 0.2), ("b.pdf", 0.1)), support=0.7) is Confidence.MEDIA
    assert (
        service._confidence(_selected(("a.pdf", service.thresholds.min_score - 0.01)), support=1.0) is Confidence.BAIXA
    )


def test_confidence_end_to_end_on_real_corpus(service: RAGService, captured: ListHandler) -> None:
    strong = service.answer("Como acompanho meu pedido?")
    assert strong.status == "answered" and strong.confidence in {Confidence.ALTA, Confidence.MEDIA}
    weak = service.answer("Quais formas de pagamento são aceitas?")  # fontes de documentos distintos, gap pequeno
    assert weak.status == "answered" and weak.confidence in {Confidence.MEDIA, Confidence.BAIXA}
    events = captured.events("query.answered")
    assert {event["confidence"] for event in events} <= {"alta", "média", "baixa"}


def test_confidence_distribution_is_not_degenerate() -> None:
    """Regressão do mutante 'confidence sempre alta': o eval local precisa produzir mais de um nível."""
    service = RAGService(ROOT / "corpus", Settings())
    levels = {service.answer(case.question).confidence for case in load_cases() if case.expects_sources is True}
    assert len(levels) >= 2


# ---------------------------------------------------------------------------
# evals/calibrate.py
# ---------------------------------------------------------------------------


def test_calibrate_sweep_reproduces_current_profile_outcome() -> None:
    service = RAGService(ROOT / "corpus", Settings())
    cases = load_cases()
    fused = {case.id: service.retriever.fuse(case.question) for case in cases}
    outcome = calibrate.evaluate_thresholds(fused, cases, THRESHOLD_PROFILES["local"], k=5)
    assert outcome.correct_refusal_rate == 1.0
    assert outcome.false_refusal_rate is not None and outcome.false_refusal_rate <= 0.05
    assert outcome.answered + outcome.refused == len(cases)
    # Um piso absurdo recusa tudo: recusa correta 100 %, indevida 100 %.
    strict = calibrate.evaluate_thresholds(
        fused, cases, dataclasses.replace(THRESHOLD_PROFILES["local"], min_score=0.99), k=5
    )
    assert strict.answered == 0 and strict.false_refusal_rate == 1.0


def test_calibrate_ranks_by_correct_then_false_refusal() -> None:
    service = RAGService(ROOT / "corpus", Settings())
    cases = load_cases()
    grid = {"min_score": [0.12, 0.99], "min_lexical_coverage": [0.2]}
    results, _ = calibrate.sweep(service, cases, grid)
    assert len(results) == 2
    best_thresholds, best_outcome = results[0]
    assert best_thresholds.min_score == 0.12 and best_outcome.false_refusal_rate <= 0.05
    assert results[0][1].key() >= results[1][1].key()


def test_calibrate_cli_prints_table_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(calibrate, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_MODE", raising=False)
    code = calibrate.main(
        [
            "--mode",
            "local",
            "--min-score",
            "0.12",
            "--vector-only",
            "0.5",
            "--vector-with-overlap",
            "0.35",
            "--coverage",
            "0.2",
            "--save",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "perfil atual" in out and "combinações avaliadas: 1" in out
    (saved,) = tmp_path.glob("*-calibration-local.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["current_profile"]["thresholds"] == dataclasses.asdict(THRESHOLD_PROFILES["local"])
    assert payload["score_distribution"]["top1_in_scope"]["median"] is not None


def test_calibrate_cli_json_and_invalid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RAG_MODE", raising=False)
    assert (
        calibrate.main(
            [
                "--json",
                "--min-score",
                "0.12",
                "--vector-only",
                "0.5",
                "--vector-with-overlap",
                "0.35",
                "--coverage",
                "0.2",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["combinations"] == 1 and payload["best"][0]["outcome"]["correct_refusal_rate"] == 1.0
    monkeypatch.setenv("RAG_TOP_K", "abc")
    assert calibrate.main(["--json"]) == 2


def test_eval_case_helper_hits_by_source_when_no_chunk_ids() -> None:
    case = EvalCase(id="x", category="in_scope", question="q", expected_sources=("faq.csv",))
    chunk = RetrievedChunk(Chunk(id="faq.csv:r2", text="t", source="faq.csv", locator={}), 0.5)
    candidate = calibrate.ScoredCandidate(chunk.chunk, 0.5, 1, 1.0, 1, 1, 1.0, 0.03)
    assert calibrate._hit(candidate, case)
    assert not calibrate._hit(
        dataclasses.replace(candidate, chunk=Chunk(id="o.pdf:p1:c1", text="t", source="o.pdf", locator={})), case
    )


def test_thresholds_dataclass_is_frozen_and_complete() -> None:
    profile = THRESHOLD_PROFILES["local"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.min_score = 0.5  # type: ignore[misc]
    assert set(dataclasses.asdict(RetrievalThresholds(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))) == {
        "min_score",
        "vector_only_min_score",
        "vector_with_overlap_min_score",
        "min_lexical_coverage",
        "high_confidence_score",
        "relative_gap",
        "mmr_lambda",
    }
