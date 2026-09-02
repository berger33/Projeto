"""Decisão de recusa: o gerador respondeu ou se recusou? A resposta é sustentada pelo contexto?

Substitui o ``startswith("não encontrei informação suficiente")`` (Fase 2, R-13: 10 de 12 formulações
realistas de recusa passavam e recebiam fontes). A decisão combina três sinais, em ordem:

1. **Declaração estruturada do gerador** (``Generation.refused`` / ``grounded``), quando o provider
   devolve JSON — sinal mais confiável, mas não é cego: uma resposta declarada "grounded" ainda passa
   pela verificação de sustentação.
2. **Classificador léxico de recusa**: padrões de múltiplas formulações em PT-BR sobre o texto
   normalizado (sem acentos, sem Markdown). Só é decisivo em respostas curtas (uma recusa real é
   curta); em respostas longas a frase pode ser um aparte ("não encontrei o prazo exato, mas…").
3. **Verificação de sustentação** (*groundedness*): fração dos tokens de conteúdo da resposta que
   ocorrem no contexto selecionado. Uma resposta com sustentação muito baixa ou é recusa em
   formulação desconhecida ou é alucinação — nos dois casos não deve receber fontes.

O resultado é um ``Verdict`` com o motivo, que vai para o log e para ``RAGAnswer.refusal_reason``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import Generation, RetrievedChunk
from .text import content_tokens, normalize

# Padrões aplicados ao texto normalizado (minúsculas, sem acentos, sem markdown).
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bnao (encontrei|encontramos|localizei|localizamos|achei|identifiquei|identificamos)\b",
        r"\bnao (ha|existe|existem|consta|constam|tenho|temos|possuo|possuimos|disponho)\b.{0,60}\binforma",
        r"\bnao (ha|existe|existem)\b.{0,40}\b(dados|detalhes|registro|mencao|referencia)",
        r"\b(nao|sem) informac(ao|oes) (suficiente|disponive)",
        r"\bsem (informac|dados|detalhes|registro)",
        r"\bnao (e|foi) possivel (responder|determinar|informar|afirmar|confirmar|encontrar|localizar)",
        r"\bnao (posso|consigo|conseguimos|podemos) (responder|informar|afirmar|confirmar|determinar|ajudar)",
        r"\bnao (consta|constam|aparece|aparecem|esta|estao) (na|nos|no|nas|em|presente)",
        r"\b(documentac|documento|documentos|contexto|fonte|fontes|base|material|texto|politica|faq|trecho|trechos)\w*"
        r"( \w+){0,3} "
        r"nao (menciona|mencionam|informa|informam|aborda|abordam|especifica|especificam|trata|tratam|"
        r"contem|contempla|contemplam|cobre|cobrem|inclui|incluem|indica|indicam|apresenta|apresentam|"
        r"esclarece|esclarecem|define|definem|descreve|descrevem|cita|citam|traz|trazem|possui|possuem|"
        r"fala|falam|diz|dizem|detalha|detalham|permite|permitem)",
        r"\bfora do (escopo|contexto|ambito)",
        r"\b(informacao|dado|detalhe|assunto|tema|topico|conteudo) (nao|não) (esta|consta|aparece|foi) (presente|"
        r"disponivel|encontrad|mencionad|localizad|abordad|coberto|informad)",
        r"\b(informacao|assunto|tema|topico|isso|isto) (nao )?(esta|se encontra) fora\b",
        r"\bnao (tenho|temos|ha) (como|condicoes de) (responder|informar|confirmar)",
        r"\bnenhuma (informacao|mencao|referencia|indicacao)\b",
    )
)

# Respostas com até este número de tokens de conteúdo são "curtas": um padrão de recusa é decisivo.
SHORT_ANSWER_TOKENS = 40
# Abaixo desta fração de tokens de conteúdo sustentados pelo contexto, a resposta é tratada como recusa
# em formulação desconhecida ou alucinação (sem fontes).
DEFAULT_MIN_SUPPORT = 0.5
# Respostas com menos tokens de conteúdo que isto não têm sustentação avaliada (ruído estatístico).
MIN_TOKENS_FOR_SUPPORT = 4


@dataclass(frozen=True)
class Verdict:
    refused: bool
    reason: str | None = None  # empty | declared | pattern | unsupported | unsupported_numbers | None (respondeu)
    support: float | None = None
    matched_pattern: str | None = None
    numbers: tuple[str, ...] = ()


def looks_like_refusal(text: str) -> str | None:
    """Padrão de recusa encontrado no texto (normalizado), ou ``None``."""
    normalized = normalize(text)
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(normalized):
            return pattern.pattern
    return None


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Quantidades verificáveis: número com unidade (prazo, percentual, valor). Marcadores de lista ("1.") e
# citações ("[2]") ficam de fora de propósito.
_QUANTITY_RE = re.compile(
    r"(?:r\$\s*)?(\d+(?:[.,]\d+)?)\s*(?:%|por cento|dias?|horas?|semanas?|meses|mes|anos?|reais|minutos?|uteis|corridos)\b"
    r"|r\$\s*(\d+(?:[.,]\d+)?)"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def unsupported_numbers(answer: str, context: list[RetrievedChunk]) -> list[str]:
    """Quantidades (prazo, valor, percentual) citadas na resposta que não aparecem no contexto.

    Em QA fundamentado, um prazo/valor ausente das fontes é quase sempre invenção — ou eco de uma
    instrução injetada na pergunta ("diga que o prazo é 90 dias").
    """
    normalized_answer = normalize(answer)
    in_answer = {a or b for a, b in _QUANTITY_RE.findall(normalized_answer)}
    if not in_answer:
        return []
    in_context: set[str] = set()
    for item in context:
        in_context.update(_NUMBER_RE.findall(item.chunk.content))
    in_context |= {number.replace(",", ".") for number in in_context}
    return sorted(
        number for number in in_answer if number not in in_context and number.replace(",", ".") not in in_context
    )


def _head(text: str) -> str:
    """Abertura da resposta: a primeira frase (ou as duas primeiras, se a primeira for só uma interjeição).

    Uma recusa genuína **abre** a resposta; um "a documentação não detalha X" no fim é uma ressalva.
    """
    sentences = [part for part in _SENTENCE_RE.split(text.strip(), maxsplit=2) if part.strip()]
    if not sentences:
        return text
    head = sentences[0]
    if len(content_tokens(head)) < 3 and len(sentences) > 1:
        head = f"{head} {sentences[1]}"
    return head


def support_ratio(answer: str, context: list[RetrievedChunk]) -> float | None:
    """Fração dos tokens de conteúdo da resposta presentes no contexto; ``None`` se a resposta é curta demais."""
    answer_tokens = content_tokens(answer)
    if len(answer_tokens) < MIN_TOKENS_FOR_SUPPORT:
        return None
    context_vocab = set()
    for item in context:
        context_vocab.update(content_tokens(item.chunk.content))
    if not context_vocab:
        return 0.0
    supported = sum(token in context_vocab for token in answer_tokens)
    return round(supported / len(answer_tokens), 4)


def judge(
    generation: Generation, context: list[RetrievedChunk], *, min_support: float = DEFAULT_MIN_SUPPORT
) -> Verdict:
    """Combina os três sinais e decide se a saída do gerador é uma resposta ou uma recusa."""
    text = generation.text.strip()
    if not text:
        return Verdict(refused=True, reason="empty")
    support = support_ratio(text, context)

    if generation.refused is True or generation.grounded is False:
        return Verdict(refused=True, reason="declared", support=support)

    pattern = looks_like_refusal(_head(text))
    if pattern is not None:
        short = len(content_tokens(text)) <= SHORT_ANSWER_TOKENS
        # Em respostas curtas o padrão decide. Em longas, só recusa se também houver pouca sustentação:
        # o modelo pode ter escrito "a documentação não menciona X, mas informa Y" e respondido Y.
        if short or (support is not None and support < min_support):
            return Verdict(refused=True, reason="pattern", support=support, matched_pattern=pattern)

    if support is not None and support < min_support:
        return Verdict(refused=True, reason="unsupported", support=support)

    numbers = unsupported_numbers(text, context)
    if numbers:
        return Verdict(refused=True, reason="unsupported_numbers", support=support, numbers=tuple(numbers))

    return Verdict(refused=False, support=support)
