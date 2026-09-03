"""Busca lexical BM25 sobre texto normalizado em PT-BR (Fase 2, R-10).

Implementação própria (~80 linhas), sem dependência externa. Opera sobre os tokens de ``app.text``
(sem acentos, minúsculas, sem stopwords) reduzidos por um *stemmer* leve — sufixos flexionais mais
comuns do português — para que ``devolucao``/``devolucoes``, ``reembolso``/``reembolsos`` e
``pagar``/``pagamento`` (parcialmente) se encontrem.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .text import content_tokens

# Sufixos flexionais/derivacionais frequentes; o mais longo que casar é removido, desde que o
# radical mantenha >= MIN_STEM caracteres. Não é o RSLP completo: o objetivo é casar plural,
# gênero, gerúndio/particípio e nominalizações comuns (devolução/devoluções, pagar/pagamento/pago).
_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            "amentos",
            "imentos",
            "amento",
            "imento",
            "acoes",
            "icoes",
            "ucoes",
            "idades",
            "idade",
            "mente",
            "acao",
            "icao",
            "ucao",
            "coes",
            "cao",
            "ncia",
            "ismo",
            "ista",
            "avel",
            "ivel",
            "oes",
            "aes",
            "ais",
            "eis",
            "ois",
            "ado",
            "ada",
            "ido",
            "ida",
            "ados",
            "adas",
            "idos",
            "idas",
            "ando",
            "endo",
            "indo",
            "aram",
            "eram",
            "iram",
            "ar",
            "er",
            "ir",
            "ao",
            "es",
            "as",
            "os",
            "s",
            "a",
            "o",
            "e",
        },
        key=len,
        reverse=True,
    )
)
MIN_STEM = 3


def stem(token: str) -> str:
    """Stemmer leve para PT-BR. Não altera números nem tokens com até 3 caracteres."""
    if token.isdigit() or len(token) <= MIN_STEM:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
            return token[: -len(suffix)]
    return token


def analyze(text: str) -> list[str]:
    """Tokens de conteúdo normalizados e reduzidos ao radical."""
    return [stem(token) for token in content_tokens(text)]


@dataclass(frozen=True)
class LexicalHit:
    index: int
    score: float


class BM25Index:
    """BM25 (Okapi) sobre uma lista de documentos já tokenizados por ``analyze``."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [analyze(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(documents)
        # IDF com suavização (Lucene): sempre positivo, evita termos muito comuns com peso negativo.
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.tokens)

    def search(self, query: str, *, k: int = 10) -> list[LexicalHit]:
        terms = analyze(query)
        if not terms or not self.tokens:
            return []
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = self.k1 * (1.0 - self.b + self.b * self.lengths[index] / (self.avg_length or 1.0))
            for term in terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                score += self.idf[term] * frequency * (self.k1 + 1.0) / (frequency + length_norm)
            scores.append(score)
        ranked = sorted(
            (LexicalHit(index=index, score=score) for index, score in enumerate(scores) if score > 0.0),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return ranked[: max(1, k)]

    def overlap(self, query: str, index: int) -> int:
        """Nº de termos (radicais) distintos da consulta presentes no documento ``index``."""
        return len(set(analyze(query)) & set(self.tokens[index]))

    def coverage(self, query: str, index: int) -> float:
        """Fração **ponderada por IDF** dos termos da consulta presentes no documento ``index``.

        Um termo raro que casa vale mais que um termo comum; termos ausentes do corpus recebem o IDF
        máximo (penalizam a cobertura: a pergunta fala de algo que a base não tem). Em [0, 1].
        """
        terms = set(analyze(query))
        if not terms or not self.tokens:
            return 0.0
        total_documents = len(self.tokens)
        max_idf = math.log(1.0 + (total_documents + 0.5) / 0.5)
        total = sum(self.idf.get(term, max_idf) for term in terms)
        present = set(self.tokens[index])
        matched = sum(self.idf[term] for term in terms if term in present)
        return round(matched / total, 4) if total else 0.0
