"""P1-03: chunking reescrito com invariantes garantidas (R-03 alto, R-02 boilerplate, G-16).

Invariantes verificadas em casos dirigidos e em textos aleatórios:
tamanho máximo respeitado, recorte exato do texto normalizado, cobertura total, sem cortes
intra-palavra (exceto palavra única maior que o máximo), sobreposição entre janelas.
"""

from __future__ import annotations

import random
import re
from itertools import pairwise
from pathlib import Path

import pytest

from app.chunking import (
    CHUNKING_VERSION,
    ChunkingConfig,
    TextSpan,
    detect_boilerplate,
    estimate_tokens,
    normalize_text,
    split_document,
    split_text,
)
from app.documents import load_chunks

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Invariantes (helper compartilhado)
# ---------------------------------------------------------------------------


def _giant_word_at(normalized: str, position: int, max_chars: int) -> bool:
    word_start = max(normalized.rfind(" ", 0, position), normalized.rfind("\n", 0, position)) + 1
    match = re.compile(r"\S*").match(normalized, word_start)
    return match is not None and match.end() - word_start > max_chars


def assert_invariants(text: str, config: ChunkingConfig, spans: list[TextSpan]) -> None:
    normalized = normalize_text(text)
    if not normalized.strip():
        assert spans == []
        return
    assert spans, "texto não vazio precisa gerar ao menos um chunk"
    previous_end = 0
    for span in spans:
        assert span.token_estimate <= config.max_tokens, (span.token_estimate, config.max_tokens)
        assert len(span.text) <= config.max_chars
        assert normalized[span.char_start : span.char_end] == span.text
        assert span.text == span.text.strip()
        if span.char_start > 0 and not normalized[span.char_start - 1].isspace():
            assert _giant_word_at(normalized, span.char_start, config.max_chars), span.text[:40]
        if span.char_end < len(normalized) and not normalized[span.char_end].isspace():
            assert _giant_word_at(normalized, span.char_end, config.max_chars), span.text[-40:]
        gap = normalized[previous_end : span.char_start] if span.char_start > previous_end else ""
        assert not gap.strip(), f"conteúdo não coberto: {gap[:80]!r}"
        previous_end = max(previous_end, span.char_end)
    assert not normalized[previous_end:].strip(), "fim do texto não coberto"


# ---------------------------------------------------------------------------
# Normalização e boilerplate
# ---------------------------------------------------------------------------


def test_normalize_joins_visual_wraps_and_keeps_paragraphs_and_headings() -> None:
    raw = "AURORA\r\n1. Objetivo\nEsta política descreve como a loja\ncoleta dados.\n\nSegundo parágrafo\ncontinua aqui.\n2. Dados\nTexto."
    normalized = normalize_text(raw)
    assert normalized.split("\n") == [
        "AURORA",
        "1. Objetivo",
        "Esta política descreve como a loja coleta dados.",
        "",
        "Segundo parágrafo continua aqui.",
        "2. Dados",
        "Texto.",
    ]


def test_normalize_undoes_end_of_line_hyphenation_and_collapses_spaces() -> None:
    assert normalize_text("A devolu-\nção  do\tproduto") == "A devolução do produto"
    assert normalize_text("   \n\n  ") == ""


def test_detect_boilerplate_finds_repeated_edge_lines_across_pages() -> None:
    pages = [
        "AURORA MODA ONLINE\nTítulo A\nconteúdo a\nDocumento fictício - rodapé.\n",
        "AURORA MODA ONLINE\nTítulo B\nconteúdo b\nDocumento fictício - rodapé.\n",
        "AURORA MODA ONLINE\nTítulo C\nconteúdo c\nDocumento fictício - rodapé.\n",
    ]
    boilerplate = detect_boilerplate(pages)
    assert boilerplate == {"aurora moda online", "documento fictício - rodapé."}
    assert detect_boilerplate(pages[:1]) == frozenset()  # uma página só não define repetição
    normalized = normalize_text(pages[0], boilerplate=boilerplate)
    assert normalized == "Título A conteúdo a" or normalized == "Título A\nconteúdo a"


def test_boilerplate_requires_same_edge_position() -> None:
    # "Contato" aparece em todas as páginas, mas no meio do texto: não é boilerplate.
    pages = ["A\nContato\nfim1", "B\nContato\nfim2", "C\nContato\nfim3"]
    assert "contato" not in detect_boilerplate(pages, edge_lines=1)


# ---------------------------------------------------------------------------
# Bugs originais (R-03) e casos dirigidos
# ---------------------------------------------------------------------------


def test_r03_large_paragraph_after_buffer_is_split() -> None:
    """Bug 1: parágrafo grande após buffer não vazio nunca caía na janela deslizante."""
    text = "Curto.\n\n" + ("palavra " * 250)
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert len(chunks) > 5


def test_r03_tail_plus_paragraph_respects_limit() -> None:
    """Bug 2: tail(overlap) + parágrafo podia exceder o limite."""
    text = ("a " * 440) + "\n\n" + ("b " * 425)
    chunks = split_text(text, chunk_size=900, overlap=120)
    assert max(len(chunk) for chunk in chunks) <= 900


def test_r03_paragraph_split_actually_triggers() -> None:
    """Bug 3: split por parágrafo nunca disparava; agora cada parágrafo grande é dividido por sentença."""
    paragraph = " ".join(f"Sentença número {i} com algum conteúdo relevante." for i in range(40))
    text = f"{paragraph}\n\n{paragraph}"
    config = ChunkingConfig(max_tokens=60, overlap_tokens=10)
    spans = split_document(text, config)
    assert_invariants(text, config, spans)
    assert len(spans) >= 6
    # Nenhum chunk termina no meio de uma sentença (todos terminam em ponto).
    assert all(span.text.endswith(".") for span in spans)


def test_no_intra_word_cuts_and_overlap_between_windows() -> None:
    words = [f"palavra{i}" for i in range(400)]
    text = " ".join(words)
    config = ChunkingConfig(max_tokens=40, overlap_tokens=8)
    spans = split_document(text, config)
    assert_invariants(text, config, spans)
    assert len(spans) > 3
    for previous, current in pairwise(spans):
        assert current.char_start < previous.char_end, "janelas consecutivas devem se sobrepor"
        shared = previous.char_end - current.char_start
        assert shared <= config.overlap_chars + len("palavra999")
        assert previous.text.split()[-1] == text[current.char_start : previous.char_end].split()[-1]


def test_single_giant_word_is_hard_cut_only_when_unavoidable() -> None:
    text = "x" * 1000
    config = ChunkingConfig(max_tokens=20, overlap_tokens=0)
    spans = split_document(text, config)
    assert_invariants(text, config, spans)
    assert "".join(span.text for span in spans) == text


def test_sections_are_detected_and_prefixed_with_document_title() -> None:
    text = (
        "Política de Reembolso\n"
        "1. Prazo para devolução\nO cliente pode devolver em até 10 dias corridos após o recebimento do pedido.\n"
        "2. Condições do produto\nO produto deve estar sem sinais de uso e com etiquetas.\n"
        "3. Contato\nEnvie e-mail ao suporte."
    )
    spans = split_document(text, ChunkingConfig(max_tokens=300, min_tokens=0))
    assert [span.section for span in spans] == [
        "Política de Reembolso / 1. Prazo para devolução",
        "Política de Reembolso / 2. Condições do produto",
        "Política de Reembolso / 3. Contato",
    ]
    assert spans[0].text.startswith("Política de Reembolso\n1. Prazo para devolução")


def test_tiny_sections_are_merged_with_neighbours() -> None:
    text = "1. A\nUm.\n2. B\nDois.\n3. C\nTrês."
    spans = split_document(text, ChunkingConfig(max_tokens=300, min_tokens=24))
    assert len(spans) == 1
    assert spans[0].section == "1. A / 2. B / 3. C"


def test_markdown_headings_open_sections() -> None:
    text = "# Guia\n\n## Entrega\nPrazo de 5 dias.\n\n## Troca\nAté 30 dias."
    spans = split_document(text, ChunkingConfig(max_tokens=300, min_tokens=0))
    assert [span.section for span in spans] == ["Guia / Entrega", "Guia / Troca"]
    assert spans[0].text.startswith("# Guia")


def test_config_validation_and_char_compatibility() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=4)
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=100, overlap_tokens=100)
    legacy = ChunkingConfig.from_chars(900, 120)
    assert legacy.max_chars <= 900 and legacy.overlap_chars <= 120
    assert estimate_tokens("") == 0 and estimate_tokens("a" * 35) == 10
    assert CHUNKING_VERSION


def test_empty_and_whitespace_inputs() -> None:
    assert split_document("") == []
    assert split_document(" \n\n \t") == []
    assert split_text("") == []


# ---------------------------------------------------------------------------
# Propriedades sobre textos aleatórios
# ---------------------------------------------------------------------------

WORDS = [
    "devolução",
    "prazo",
    "dias",
    "corridos",
    "produto",
    "pedido",
    "suporte",
    "e-mail",
    "pagamento",
    "cartão",
    "PIX",
    "entrega",
    "CEP",
    "política",
    "dados",
    "cliente",
    "loja",
    "Aurora",
    "análise",
    "etiqueta",
    "embalagem",
    "reembolso",
    "banco",
]


def _random_text(rng: random.Random) -> str:
    parts: list[str] = []
    for _ in range(rng.randint(1, 10)):
        kind = rng.random()
        if kind < 0.2:
            parts.append(f"{rng.randint(1, 9)}. {' '.join(rng.choices(WORDS, k=rng.randint(1, 5))).capitalize()}")
        elif kind < 0.3:
            parts.append("x" * rng.randint(1, 1500))
        else:
            sentences = []
            for _ in range(rng.randint(1, 12)):
                sentences.append(" ".join(rng.choices(WORDS, k=rng.randint(1, 30))).capitalize() + rng.choice(".!?"))
            paragraph = " ".join(sentences)
            if rng.random() < 0.5:
                paragraph = re.sub(r"(.{40,80}) ", lambda m: m.group(1) + "\n", paragraph)
            parts.append(paragraph)
        parts.append(rng.choice(["\n", "\n\n", "\n\n\n", "\r\n"]))
    return "".join(parts)


@pytest.mark.parametrize("seed", range(12))
def test_invariants_hold_on_random_texts(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 — gerador de casos de teste, não criptografia
    for _ in range(40):
        max_tokens = rng.choice([8, 12, 20, 40, 80, 300])
        config = ChunkingConfig(
            max_tokens=max_tokens,
            overlap_tokens=min(rng.choice([0, 2, 5, 10]), max_tokens - 1),
            min_tokens=rng.choice([0, 4, 24]),
        )
        text = _random_text(rng)
        assert_invariants(text, config, split_document(text, config))


# ---------------------------------------------------------------------------
# Corpus real
# ---------------------------------------------------------------------------


def test_real_corpus_yields_one_chunk_per_section_without_boilerplate() -> None:
    chunks = load_chunks(ROOT / "docs")
    pdf_chunks = [chunk for chunk in chunks if chunk.source.endswith(".pdf")]
    assert len(pdf_chunks) == 19  # 7 FAQ + 6 privacidade + 6 reembolso
    for chunk in pdf_chunks:
        assert chunk.section and chunk.token_estimate and chunk.token_estimate <= 300
        assert chunk.char_start is not None and chunk.char_end is not None
        assert "AURORA MODA ONLINE" not in chunk.text
        assert "Documento corporativo fictício" not in chunk.text
        assert not re.search(r"\w-\n\w", chunk.text)  # sem hifenização de quebra de linha
    sections = [chunk.section for chunk in pdf_chunks if chunk.source == "politica_privacidade.pdf"]
    assert sections[1] == "Política de Privacidade / 2. Dados coletados"
    assert any(chunk.section == "Política de Reembolso e Devoluções / 6. Contato" for chunk in pdf_chunks)
    # Chunks de CSV não têm metadados de seção (uma linha = um chunk).
    assert all(chunk.section is None for chunk in chunks if chunk.source.endswith(".csv"))
