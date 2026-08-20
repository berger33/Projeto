from pathlib import Path

from app.agent import FashionStoreAgent
from app.main import health

ROOT = Path(__file__).resolve().parents[1]
agent = FashionStoreAgent(ROOT / "docs")


def test_return_window():
    assert "10 dias" in agent.answer("Qual o prazo para devolução?")["answer"]


def test_payment():
    answer = agent.answer("Quais pagamentos vocês aceitam?")["answer"].lower()
    assert "pix" in answer and "cartão" in answer


def test_privacy():
    assert "dados" in agent.answer("Como meus dados são usados?")["answer"].lower()


def test_out_of_scope():
    assert "não encontrei" in agent.answer("Qual é a capital da Austrália?")["answer"].lower()


def test_health_endpoint():
    result = health()
    assert result["status"] == "ok"
    assert result["documents"] > 0
    assert result["chunks"] > 0
    assert isinstance(result["langchain_available"], bool)
