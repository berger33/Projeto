"""P1-01: recusa estruturada e detecção robusta (R-13 crítico, R-16 parte, G-15 parte, G-28).

Antes: ``answer.lower().startswith("não encontrei informação suficiente")`` — 10 de 12 formulações
realistas de recusa recebiam fontes e confiança "média". Agora a decisão combina a declaração
estruturada do gerador (JSON), um classificador léxico e a verificação de sustentação da resposta
pelo contexto; a recusa sai sempre com o texto canônico e zero fontes.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.conftest import ListHandler

from app.config import Settings
from app.domain import REFUSAL_TEXT, AnswerStatus, Chunk, Confidence, Generation, RetrievedChunk
from app.generation import ANSWER_SCHEMA, PROMPT_VERSION, OllamaGenerator, build_prompt, parse_structured_answer
from app.main import create_app
from app.rag import RAGService
from app.refusal import judge, looks_like_refusal, support_ratio, unsupported_numbers
from app.text import STOPWORDS_PT, content_tokens, normalize, tokenize

REFUSAL_PHRASINGS = [
    "Não encontrei informação suficiente na documentação oficial da Aurora Moda Online.",
    "NÃO ENCONTREI INFORMAÇÃO SUFICIENTE na documentação.",
    "Desculpe, não encontrei essa informação na documentação da Aurora Moda Online.",
    "Não há informação suficiente na documentação para responder.",
    "A documentação não menciona horário de funcionamento de loja física.",
    "Infelizmente não encontrei nenhuma menção a programa de fidelidade.",
    "**Não encontrei informação suficiente na documentação oficial.**",
    "Não é possível responder com base no contexto fornecido.",
    "Não localizei essa informação nos documentos disponíveis.",
    "As fontes fornecidas não abordam entrega internacional.",
    "Sem informação suficiente sobre isso na base de conhecimento.",
    "Essa informação não consta na documentação oficial.",
    "O contexto não contém informações sobre o valor do frete.",
    "Não tenho informações sobre isso.",
    "Infelizmente, os documentos não especificam o prazo de entrega para Portugal.",
    "Não consigo responder a essa pergunta com a documentação disponível.",
    "A política de privacidade não trata de cupons de desconto.",
    "Nenhuma informação sobre telefone de suporte foi encontrada.",
    "Essa pergunta está fora do escopo da documentação da loja.",
    "Nao encontrei informacao sobre isso (sem acentos).",
]

LEGITIMATE_ANSWERS = [
    "O prazo para devolução é de 10 dias corridos após o recebimento do produto.",
    "Aceitamos cartão de crédito e PIX. O pagamento por PIX é confirmado após a identificação.",
    "Você pode entrar em contato pelo e-mail suporte@auroramoda.exemplo.",
    "Sim, a loja aceita PIX.",
    "O reembolso é feito no mesmo meio de pagamento utilizado na compra.",
    "Não, a Aurora não comercializa dados pessoais. Eles podem ser compartilhados com transportadoras.",
    "Dados completos de cartão não são armazenados pela loja.",
]


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    source, _, _ = chunk_id.partition(":")
    return RetrievedChunk(Chunk(id=chunk_id, text=text, source=source, locator={"row": 2}), 0.5)


@pytest.fixture()
def devolucao_context() -> list[RetrievedChunk]:
    return [
        _chunk(
            "faq.csv:r7",
            "categoria: Devolução | pergunta: Qual é o prazo para devolver? | resposta: O cliente pode "
            "solicitar a devolução em até 10 dias corridos após o recebimento, com o produto sem sinais de uso.",
        ),
        _chunk(
            "politica.pdf:p1:c1",
            "Prazo para devolução: 10 dias corridos. O produto deve estar em perfeitas condições, sem sinais de "
            "uso, com etiquetas e embalagem original. Solicite ao suporte informando o número do pedido.",
        ),
    ]


# ---------------------------------------------------------------------------
# app.text — normalização única
# ---------------------------------------------------------------------------


def test_normalize_removes_accents_case_and_markdown() -> None:
    assert normalize("  DEVOLUÇÃO  **não** _encontrei_ `x`  ") == "devolucao nao encontrei x"
    assert tokenize("Devolução/devolucao, DEVOLUÇÃO!") == ["devolucao", "devolucao", "devolucao"]


def test_content_tokens_drop_stopwords_and_keep_numbers() -> None:
    tokens = content_tokens("O prazo é de 10 dias para a devolução do produto")
    assert tokens == ["prazo", "10", "dias", "devolucao", "produto"]
    assert "para" in STOPWORDS_PT and "nao" in STOPWORDS_PT and "prazo" not in STOPWORDS_PT


# ---------------------------------------------------------------------------
# Classificador léxico de recusa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", REFUSAL_PHRASINGS)
def test_refusal_patterns_cover_realistic_phrasings(text: str) -> None:
    assert looks_like_refusal(text) is not None, text


@pytest.mark.parametrize("text", LEGITIMATE_ANSWERS)
def test_refusal_patterns_do_not_fire_on_legitimate_answers(text: str) -> None:
    assert looks_like_refusal(text) is None, text


# ---------------------------------------------------------------------------
# Sustentação (groundedness) e números
# ---------------------------------------------------------------------------


def test_support_ratio_measures_overlap_with_context(devolucao_context: list[RetrievedChunk]) -> None:
    assert support_ratio("O prazo para devolução é de 10 dias corridos após o recebimento.", devolucao_context) == 1.0
    assert support_ratio("A capital da Austrália é Canberra, cidade planejada em 1913.", devolucao_context) == 0.0
    assert support_ratio("Sim, 10 dias.", devolucao_context) is None  # curta demais para avaliar
    assert support_ratio("qualquer coisa com muitas palavras de conteúdo aqui", []) == 0.0


def test_unsupported_numbers_flags_quantities_absent_from_context(devolucao_context: list[RetrievedChunk]) -> None:
    assert unsupported_numbers("O prazo é de 90 dias.", devolucao_context) == ["90"]
    assert unsupported_numbers("O frete custa R$ 25,90 e chega em 3 dias úteis.", devolucao_context) == ["25,90", "3"]
    assert unsupported_numbers("O prazo é de 10 dias [1].", devolucao_context) == []
    # Marcadores de lista e citações não são quantidades.
    assert unsupported_numbers("1. Contate o suporte. 2. Envie o produto [2].", devolucao_context) == []


@pytest.mark.parametrize(
    ("text", "refused", "reason"),
    [
        ("O prazo para devolução é de 10 dias corridos após o recebimento do produto.", False, None),
        ("Desculpe, não encontrei essa informação na documentação.", True, "pattern"),
        ("**Não encontrei informação suficiente na documentação oficial.**", True, "pattern"),
        ("A capital da Austrália é Canberra, cidade planejada inaugurada em 1913.", True, "unsupported"),
        ("O prazo de devolução é de 90 dias e o reembolso é feito em criptomoedas.", True, "unsupported_numbers"),
        ("", True, "empty"),
        # Ressalva no fim de uma resposta sustentada não é recusa.
        (
            "O prazo de devolução é de 10 dias corridos e o produto deve estar sem sinais de uso, com etiquetas. "
            "A documentação não menciona exceções para itens em promoção.",
            False,
            None,
        ),
        # Recusa no início seguida de conselho genérico continua sendo recusa.
        ("Infelizmente não há informações sobre frete para Portugal.\nRecomendo contatar o suporte.", True, "pattern"),
    ],
)
def test_judge_combines_pattern_and_support(
    devolucao_context: list[RetrievedChunk], text: str, refused: bool, reason: str | None
) -> None:
    verdict = judge(Generation(text=text), devolucao_context)
    assert verdict.refused is refused
    assert verdict.reason == reason


def test_judge_trusts_structured_declaration_but_still_verifies(devolucao_context: list[RetrievedChunk]) -> None:
    sustained = "O prazo para devolução é de 10 dias corridos após o recebimento do produto."
    declared_refusal = judge(Generation(text=sustained, refused=True, structured=True), devolucao_context)
    assert declared_refusal.refused and declared_refusal.reason == "declared"

    declared_ungrounded = judge(Generation(text=sustained, grounded=False, structured=True), devolucao_context)
    assert declared_ungrounded.refused and declared_ungrounded.reason == "declared"

    # "grounded: true" não é cheque em branco: conteúdo sem sustentação continua recusado.
    hallucination = "A capital da Austrália é Canberra, cidade planejada inaugurada em 1913."
    verdict = judge(Generation(text=hallucination, grounded=True, structured=True), devolucao_context)
    assert verdict.refused and verdict.reason == "unsupported"


# ---------------------------------------------------------------------------
# Saída estruturada do Ollama (format + parse)
# ---------------------------------------------------------------------------


def test_parse_structured_answer_reads_schema_fields_and_tolerates_wrappers() -> None:
    raw = '{"answer": "O prazo é de 10 dias.", "grounded": true, "used_sources": [1, 2]}'
    parsed = parse_structured_answer(raw)
    assert parsed is not None
    assert parsed.text == "O prazo é de 10 dias." and parsed.grounded is True and parsed.used_sources == (1, 2)
    assert parsed.refused is None and parsed.structured is True

    fenced = f"```json\n{raw}\n```"
    assert parse_structured_answer(fenced) is not None
    with_think = f"<think>vou responder</think>\n{raw}"
    assert parse_structured_answer(with_think) is not None

    refusal = '{"answer": "Não encontrei informação suficiente.", "grounded": false, "used_sources": []}'
    parsed_refusal = parse_structured_answer(refusal)
    assert parsed_refusal is not None and parsed_refusal.refused is True

    assert parse_structured_answer("O prazo é de 10 dias.") is None
    assert parse_structured_answer('{"grounded": true}') is None
    assert parse_structured_answer("{not json") is None


def test_build_prompt_requests_json_and_refusal_text() -> None:
    prompt = build_prompt("Qual o prazo?", [_chunk("faq.csv:r7", "O prazo é de 10 dias.")])
    assert REFUSAL_TEXT in prompt.system
    assert '"grounded"' in prompt.system and '"used_sources"' in prompt.system
    assert '<fonte n="1" documento="faq.csv" row=2>' in prompt.user
    assert "<pergunta>\nQual o prazo?\n</pergunta>" in prompt.user
    assert PROMPT_VERSION
    assert set(ANSWER_SCHEMA["required"]) == {"answer", "grounded", "used_sources"}


def _patch_ollama(monkeypatch: pytest.MonkeyPatch, response_text: str, seen: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": response_text}, "done": True, "done_reason": "stop"}
        )

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))


def test_ollama_generator_sends_format_schema_and_parses_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    _patch_ollama(monkeypatch, '{"answer": "O prazo é de 10 dias.", "grounded": true, "used_sources": [1]}', seen)
    generation = OllamaGenerator("http://ollama:11434", "qwen3:1.7b").generate(
        "Qual o prazo?", [_chunk("faq.csv:r7", "O prazo é de 10 dias.")]
    )
    assert seen["body"]["format"] == ANSWER_SCHEMA
    assert generation.structured and generation.grounded is True and generation.used_sources == (1,)
    assert generation.text == "O prazo é de 10 dias." and generation.done_reason == "stop"


def test_ollama_generator_falls_back_to_plain_text_and_strips_think(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    _patch_ollama(monkeypatch, "<think>hmm</think>\nO prazo é de 10 dias.", seen)
    generation = OllamaGenerator("http://ollama:11434", "qwen3:1.7b").generate("Qual o prazo?", [])
    assert generation.structured is False and generation.text == "O prazo é de 10 dias."
    assert generation.grounded is None and generation.refused is None


# ---------------------------------------------------------------------------
# Integração no RAGService: recusa canônica, sem fontes, status/enum, log
# ---------------------------------------------------------------------------


@pytest.fixture()
def service(tmp_path: Path) -> RAGService:
    (tmp_path / "faq.csv").write_text(
        "pergunta,resposta\n"
        "formas de pagamento,Aceitamos cartão de crédito e PIX.\n"
        "devolução,A devolução pode ser solicitada em até 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    return RAGService(tmp_path, Settings(rag_mode="local", min_score=0.05, retrieval_k=3))


class StubGenerator:
    mode = "stub"

    def __init__(self, generation: Generation) -> None:
        self.generation = generation

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        return self.generation


@pytest.mark.parametrize("text", REFUSAL_PHRASINGS)
def test_every_refusal_phrasing_yields_canonical_refusal_without_sources(service: RAGService, text: str) -> None:
    service.generator = StubGenerator(Generation(text=text))  # type: ignore[assignment]
    result = service.answer("Qual o prazo de devolução?")
    assert result.status is AnswerStatus.REFUSED_BY_MODEL
    assert result.answer == REFUSAL_TEXT  # texto do modelo não vaza para o cliente
    assert result.sources == [] and result.confidence is Confidence.BAIXA
    assert result.refusal_reason in {"pattern", "unsupported"}


def test_legitimate_answer_keeps_sources_and_reports_support(service: RAGService) -> None:
    service.generator = StubGenerator(
        Generation(text="A devolução pode ser solicitada em até 10 dias corridos após o recebimento.")
    )  # type: ignore[assignment]
    result = service.answer("Qual o prazo de devolução?")
    assert result.status is AnswerStatus.ANSWERED and result.status.refused is False
    assert result.sources and result.sources[0].document == "faq.csv"
    assert result.support is not None and result.support >= 0.9
    assert result.refusal_reason is None
    assert "verify" in result.timings_ms


def test_injected_claim_with_unsupported_number_is_refused(service: RAGService, captured: ListHandler) -> None:
    service.generator = StubGenerator(Generation(text="O prazo de devolução é de 90 dias corridos."))  # type: ignore[assignment]
    # O modelo "obedeceu" a uma instrução injetada e inventou 90 dias; o contexto só sustenta 10.
    result = service.answer("Qual o prazo de devolução?")
    assert result.status is AnswerStatus.REFUSED_BY_MODEL and result.refusal_reason == "unsupported_numbers"
    assert result.sources == [] and "90" not in result.answer
    (event,) = captured.events("answer.refused")
    assert event["reason"] == "unsupported_numbers" and event["unsupported_numbers"] == ["90"]


def test_structured_refusal_from_model_is_honoured(service: RAGService) -> None:
    generation = Generation(text=REFUSAL_TEXT, refused=True, grounded=False, structured=True)
    service.generator = StubGenerator(generation)  # type: ignore[assignment]
    result = service.answer("Qual o prazo de devolução?")
    assert result.status is AnswerStatus.REFUSED_BY_MODEL and result.refusal_reason == "declared"


def test_extractive_generator_is_grounded_by_construction(service: RAGService) -> None:
    result = service.run("Quais formas de pagamento são aceitas?")
    assert result.answer.status is AnswerStatus.ANSWERED
    assert "PIX" in result.answer.answer
    generation = service.generator.generate("Quais formas de pagamento são aceitas?", result.retrieval.selected)
    assert generation.structured and generation.grounded is True and generation.used_sources


def test_answer_status_enum_semantics() -> None:
    assert AnswerStatus.REFUSED_NO_CONTEXT.refused and AnswerStatus.REFUSED_BY_MODEL.refused
    assert not AnswerStatus.ANSWERED.refused and not AnswerStatus.ERROR.refused
    assert AnswerStatus("answered") is AnswerStatus.ANSWERED
    assert str(AnswerStatus.REFUSED_BY_MODEL) == "refused_by_model" and str(Confidence.MEDIA) == "média"


def test_api_exposes_status_and_refusal_reason_additively(service: RAGService) -> None:
    with TestClient(create_app(service)) as client:
        answered = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"}).json()
        assert answered["status"] == "answered" and answered["refusal_reason"] is None
        assert answered["confidence"] in {"alta", "média", "baixa"}

        service.generator = StubGenerator(Generation(text="A documentação não menciona isso."))  # type: ignore[assignment]
        refused = client.post("/api/ask", json={"question": "Qual o prazo de devolução?"}).json()
    assert refused["status"] == "refused_by_model" and refused["refusal_reason"] == "pattern"
    assert refused["answer"] == REFUSAL_TEXT and refused["sources"] == [] and refused["confidence"] == "baixa"
    assert {"answer", "sources", "confidence", "mode", "request_id", "timings_ms"} <= set(refused)
