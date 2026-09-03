"""Derivação das fontes citadas a partir do uso real pelo gerador (Fase 2, R-17/R-16).

Ordem de preferência:

1. ``Generation.used_sources`` — índices ``[n]`` das ``<fonte n>`` declarados na saída JSON;
2. marcadores ``[n]`` / ``[n, m]`` / ``[FONTE n]`` escritos no texto da resposta;
3. **fallback**: chunks selecionados (todos, na ordem do retrieval), marcados ``inferred=True``.

Índices fora da faixa são ignorados; se nenhum índice válido sobrar, cai no fallback. Em todos os
casos, cada fonte carrega ``chunk_id``, ``score``, ``section`` e um ``excerpt`` — a frase do chunk
com maior sobreposição lexical com a resposta (ou o início do chunk).
"""

from __future__ import annotations

import re

from .domain import Generation, RetrievedChunk, SourceRef
from .lexical import analyze

EXCERPT_MAX_CHARS = 200
_CITATION_RE = re.compile(r"\[\s*(?:fonte\s*)?(\d+(?:\s*[,;]\s*\d+)*)\s*\]", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def cited_indexes(text: str) -> list[int]:
    """Índices 1-based citados como ``[1]``, ``[1, 3]`` ou ``[FONTE 2]`` no texto, na ordem, sem repetição."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def strip_citations(text: str) -> str:
    """Remove marcadores ``[n]`` do texto (a API expõe as fontes estruturadas, não inline)."""
    cleaned = _CITATION_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def best_excerpt(chunk_text: str, answer: str, *, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Frase do chunk com maior sobreposição de radicais com a resposta; empate → a primeira."""
    answer_terms = set(analyze(answer))
    sentences = [part.strip() for part in _SENTENCE_RE.split(chunk_text) if part.strip()]
    if not sentences:
        return _clip(chunk_text, max_chars)
    best = max(sentences, key=lambda sentence: (len(answer_terms & set(analyze(sentence))), -sentences.index(sentence)))
    return _clip(best, max_chars)


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _ref(item: RetrievedChunk, answer: str, *, inferred: bool) -> SourceRef:
    return SourceRef(
        document=item.chunk.source,
        page=item.chunk.locator.get("page"),
        row=item.chunk.locator.get("row"),
        chunk_id=item.chunk.id,
        score=round(item.score, 4),
        section=item.chunk.section,
        excerpt=best_excerpt(item.chunk.content, answer),
        inferred=inferred,
    )


def derive_sources(generation: Generation, selected: list[RetrievedChunk]) -> tuple[list[SourceRef], str]:
    """``(fontes, motivo)`` — motivo em {"structured", "inline", "fallback"}."""
    if not selected:
        return [], "fallback"
    declared = [n for n in generation.used_sources if 1 <= n <= len(selected)]
    reason = "structured"
    if not declared:
        declared = [n for n in cited_indexes(generation.text) if 1 <= n <= len(selected)]
        reason = "inline"
    if not declared:
        return [_ref(item, generation.text, inferred=True) for item in selected], "fallback"
    refs: list[SourceRef] = []
    for number in declared:
        ref = _ref(selected[number - 1], generation.text, inferred=False)
        if ref.chunk_id not in {existing.chunk_id for existing in refs}:
            refs.append(ref)
    return refs, reason
