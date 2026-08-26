from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.rag import RAGService


@pytest.fixture()
def eval_service(tmp_path: Path) -> RAGService:
    (tmp_path / "base.csv").write_text(
        "tema,conteudo\n"
        "pagamento,A Aurora aceita PIX e cartão de crédito.\n"
        "devolução,O prazo de devolução é de 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    return RAGService(tmp_path, Settings(rag_mode="local", min_score=0.05, retrieval_k=2))


def test_portfolio_eval_contract(eval_service: RAGService):
    cases = json.loads((Path(__file__).parents[1] / "evals" / "cases.json").read_text(encoding="utf-8"))
    for case in cases:
        result = eval_service.answer(case["question"])
        lowered = result.answer.lower()
        for fragment in case["must_contain"]:
            assert fragment.lower() in lowered, case["id"]
        assert bool(result.sources) is bool(case["expects_sources"]), case["id"]
