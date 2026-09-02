"""Chunking de texto com invariantes garantidas (Fase 2, R-03/R-02/G-16).

Pipeline por página/documento:

1. **Normalização** (``normalize_text``): ``\\r\\n`` → ``\\n``; espaços/tabs colapsados; hifenização de fim de
   linha desfeita (``devolu-\\nção`` → ``devolução``); quebras de linha de *wrap* visual (PDF) viram espaço
   quando a linha seguinte continua a frase; linhas de boilerplate (cabeçalho/rodapé repetidos) removidas.
2. **Seções**: títulos numerados (``2. Dados coletados``) e Markdown (``## Título``) abrem uma seção; o
   texto antes do primeiro título fica na seção ``None`` (ou no título do documento, se houver).
3. **Split hierárquico** dentro de cada seção que exceda o orçamento: parágrafo → sentença → janela de
   palavras com sobreposição. Nunca corta dentro de uma palavra (exceto se **uma única palavra** exceder o
   máximo — caso degenerado, ex.: URL gigante — quando o corte é inevitável e sinalizado no log).
4. **Empacotamento**: unidades consecutivas da mesma seção são reunidas até o máximo; seções muito
   pequenas (``min_tokens``) são anexadas à vizinha para não gerar chunks sem conteúdo.

Invariantes (testadas em ``tests/test_chunking.py``):

- todo chunk tem ``token_estimate <= max_tokens``;
- todo chunk é um recorte exato do texto normalizado: ``normalized[char_start:char_end] == text``;
- a união dos chunks cobre todo o texto normalizado (nada é descartado além de boilerplate/espaços);
- chunks consecutivos oriundos de janelas se sobrepõem em ~``overlap_tokens``;
- nenhum chunk começa ou termina no meio de uma palavra.

Tokens são **estimados** por caracteres (``chars_per_token``, padrão 3,5 para PT-BR); a estimativa é
deliberadamente conservadora para que o orçamento de contexto do gerador (P1-05) não estoure.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

CHUNKING_VERSION = "2"  # muda quando o algoritmo mudar (invalida índices persistidos, P2-03)

_HEADING_NUMBERED = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)[.)]\s+(\S.{0,118})$")
_HEADING_MARKDOWN = re.compile(r"^\s*#{1,6}\s+(\S.{0,118})$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"“(\[A-ZÀ-ÝÇ0-9])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_HYPHEN_WRAP = re.compile(r"(?<=[a-záéíóúâêôãõç])-\n(?=[a-záéíóúâêôãõç])")
_SPACES = re.compile(r"[ \t\f\v]+")


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 300
    overlap_tokens: int = 45
    min_tokens: int = 24
    chars_per_token: float = 3.5

    def __post_init__(self) -> None:
        if self.max_tokens < 8:
            raise ValueError("max_tokens deve ser >= 8")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens deve estar em [0, max_tokens)")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token deve ser positivo")

    @classmethod
    def from_chars(cls, chunk_size: int, overlap: int, *, chars_per_token: float = 3.5) -> ChunkingConfig:
        """Compatibilidade com a API antiga baseada em caracteres."""
        max_tokens = max(8, math.floor(chunk_size / chars_per_token))
        overlap_tokens = min(max_tokens - 1, max(0, math.floor(overlap / chars_per_token)))
        return cls(max_tokens=max_tokens, overlap_tokens=overlap_tokens, chars_per_token=chars_per_token)

    @property
    def max_chars(self) -> int:
        return math.floor(self.max_tokens * self.chars_per_token)

    @property
    def overlap_chars(self) -> int:
        return math.floor(self.overlap_tokens * self.chars_per_token)

    @property
    def min_chars(self) -> int:
        return math.floor(self.min_tokens * self.chars_per_token)


@dataclass(frozen=True)
class TextSpan:
    """Um chunk ainda sem identidade de documento: texto + posição no texto normalizado."""

    text: str
    char_start: int
    char_end: int
    section: str | None
    token_estimate: int


def estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    return math.ceil(len(text) / chars_per_token) if text else 0


# ---------------------------------------------------------------------------
# Normalização e boilerplate
# ---------------------------------------------------------------------------


def _is_heading(line: str) -> bool:
    return bool(_HEADING_NUMBERED.match(line) or _HEADING_MARKDOWN.match(line))


def _heading_title(line: str) -> str:
    match = _HEADING_NUMBERED.match(line)
    if match:
        return f"{match.group(1)}. {match.group(2).strip()}"
    match = _HEADING_MARKDOWN.match(line)
    return match.group(1).strip() if match else line.strip()


def normalize_line_key(line: str) -> str:
    return _SPACES.sub(" ", line).strip().lower()


def detect_boilerplate(pages: Iterable[str], *, edge_lines: int = 2, min_pages: int = 2) -> frozenset[str]:
    """Linhas de cabeçalho/rodapé: repetem-se na **mesma posição de borda** (ex.: 1ª linha, última linha)
    em pelo menos ``min_pages`` páginas e em metade ou mais das páginas analisadas.

    Pode ser aplicado a todas as páginas de um corpus: o papel timbrado de uma empresa se repete
    entre documentos, não só entre páginas.
    """
    counter: Counter[tuple[str, str]] = Counter()
    total = 0
    for page in pages:
        lines = [normalize_line_key(line) for line in page.replace("\r\n", "\n").split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue
        total += 1
        positions: set[tuple[str, str]] = set()
        for offset in range(min(edge_lines, len(lines))):
            positions.add((f"top{offset}", lines[offset]))
            positions.add((f"bottom{offset}", lines[-1 - offset]))
        counter.update(positions)
    threshold = max(min_pages, math.ceil(total / 2))
    return frozenset(line for (_, line), count in counter.items() if count >= threshold and len(line) >= 4)


def normalize_text(text: str, *, boilerplate: frozenset[str] = frozenset()) -> str:
    """Texto limpo e com quebras de linha significativas (ver docstring do módulo)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP.sub("", text)
    raw_lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in raw_lines if normalize_line_key(line) not in boilerplate or not line]

    joined: list[str] = []
    for line in lines:
        if not line:
            if joined and joined[-1] != "":
                joined.append("")
            continue
        if joined and joined[-1] != "" and not _is_heading(line) and not _is_heading(joined[-1]):
            previous = joined[-1]
            continues = line[0].islower() or not previous.endswith((".", "!", "?", ":", ";"))
            if continues:
                joined[-1] = f"{previous} {line}"
                continue
        joined.append(line)
    while joined and joined[-1] == "":
        joined.pop()
    while joined and joined[0] == "":
        joined.pop(0)
    return "\n".join(joined)


# ---------------------------------------------------------------------------
# Split hierárquico
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    section: str | None


def _document_title(normalized: str) -> str | None:
    """Título do documento: uma única linha curta sem pontuação final antes do primeiro título de
    seção, ou um título Markdown de nível 1 no início (``# Guia``) seguido de subtítulos."""
    lines = [line for line in normalized.split("\n") if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if _HEADING_MARKDOWN.match(first) and first.lstrip().startswith("# ") and len(lines) > 1:
        return _heading_title(first)
    first_heading = next((i for i, line in enumerate(lines) if _is_heading(line)), None)
    if first_heading is None:
        return None
    preamble = lines[:first_heading]
    if len(preamble) == 1 and len(preamble[0]) <= 100 and not preamble[0].endswith((".", "!", "?", ":", ";")):
        return preamble[0].strip()
    return None


def _sections(normalized: str) -> list[tuple[str | None, int, int]]:
    """``(título, início, fim)`` de cada seção, cobrindo o texto inteiro em ordem.

    Títulos de seção são prefixados pelo título do documento quando detectado
    (``"Política de Privacidade / 2. Dados coletados"``), o que desambigua seções homônimas
    entre documentos (``"6. Contato"``). A linha-título do documento (e qualquer preâmbulo) fica
    junto da primeira seção real, sem gerar uma seção própria.
    """
    document_title = _document_title(normalized)
    sections: list[tuple[str | None, int, int]] = []
    position = 0
    current_title: str | None = document_title
    current_start = 0
    opened = False  # já existe uma seção real (com título de seção) em aberto?
    for line in normalized.split("\n"):
        if _is_heading(line):
            heading = _heading_title(line)
            is_document_title_line = document_title is not None and not opened and heading == document_title
            if not is_document_title_line:
                if opened and position > current_start and normalized[current_start:position].strip():
                    sections.append((current_title, current_start, position))
                    current_start = position
                current_title = f"{document_title} / {heading}" if document_title else heading
                opened = True
        position += len(line) + 1
    if normalized[current_start:].strip():
        sections.append((current_title, current_start, len(normalized)))
    return sections


def _split_by(pattern: re.Pattern[str], normalized: str, start: int, end: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(normalized, start, end):
        if match.start() > cursor:
            pieces.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        pieces.append((cursor, end))
    return [(a, b) for a, b in pieces if normalized[a:b].strip()]


def _words(normalized: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", normalized[start:end])]


def _windows(normalized: str, start: int, end: int, config: ChunkingConfig) -> list[tuple[int, int]]:
    """Janelas de palavras com sobreposição; corta uma palavra só se ela sozinha exceder o máximo."""
    words = _words(normalized, start, end)
    if not words:
        return []
    windows: list[tuple[int, int]] = []
    index = 0
    while index < len(words):
        window_start = words[index][0]
        last = index
        while last + 1 < len(words) and words[last + 1][1] - window_start <= config.max_chars:
            last += 1
        if words[last][1] - window_start > config.max_chars:
            # Palavra única maior que o máximo: corte forçado em caracteres.
            hard_end = window_start + config.max_chars
            windows.append((window_start, hard_end))
            words[index] = (hard_end, words[index][1])
            continue
        windows.append((window_start, words[last][1]))
        if last + 1 >= len(words):
            break
        # Próxima janela recua até ``overlap_chars`` em palavras inteiras (sempre avança ao menos 1 palavra).
        next_index = last + 1
        while next_index - 1 > index and words[last][1] - words[next_index - 1][0] <= config.overlap_chars:
            next_index -= 1
        index = max(next_index, index + 1)
    return windows


def _split_span(normalized: str, start: int, end: int, config: ChunkingConfig) -> list[tuple[int, int]]:
    """Parágrafo → sentença → janela, só descendo de nível quando a unidade excede o máximo.

    As peças resultantes são empacotadas até o máximo com sobreposição de ``overlap_chars``.
    """
    if end - start <= config.max_chars:
        return [(start, end)]
    paragraphs = _split_by(_PARAGRAPH_BREAK, normalized, start, end)
    if len(paragraphs) > 1:
        pieces: list[tuple[int, int]] = []
        for a, b in paragraphs:
            pieces.extend(_split_span(normalized, a, b, config))
        return _pack_with_overlap(pieces, config)
    sentences = _split_by(_SENTENCE_END, normalized, start, end)
    if len(sentences) > 1:
        pieces = []
        for a, b in sentences:
            pieces.extend([(a, b)] if b - a <= config.max_chars else _windows(normalized, a, b, config))
        return _pack_with_overlap(pieces, config)
    return _windows(normalized, start, end, config)


def _pack_with_overlap(pieces: list[tuple[int, int]], config: ChunkingConfig) -> list[tuple[int, int]]:
    """Agrupa peças consecutivas (cada uma ≤ máximo) até o máximo; o grupo seguinte reaproveita as
    últimas peças (até ``overlap_chars``) para manter contexto entre chunks."""
    if len(pieces) <= 1:
        return pieces
    packed: list[tuple[int, int]] = []
    index = 0
    while index < len(pieces):
        start = pieces[index][0]
        last = index
        while last + 1 < len(pieces) and pieces[last + 1][1] - start <= config.max_chars:
            last += 1
        packed.append((start, pieces[last][1]))
        if last + 1 >= len(pieces):
            break
        next_index = last + 1
        while next_index - 1 > index and pieces[last][1] - pieces[next_index - 1][0] <= config.overlap_chars:
            next_index -= 1
        index = max(next_index, index + 1)
    return packed


def _trim(normalized: str, start: int, end: int) -> tuple[int, int]:
    while start < end and normalized[start].isspace():
        start += 1
    while end > start and normalized[end - 1].isspace():
        end -= 1
    return start, end


def split_document(
    text: str, config: ChunkingConfig | None = None, *, boilerplate: frozenset[str] = frozenset()
) -> list[TextSpan]:
    """Divide um texto (uma página de PDF, um arquivo .md/.txt) em ``TextSpan`` com as invariantes do módulo."""
    config = config or ChunkingConfig()
    normalized = normalize_text(text, boilerplate=boilerplate)
    if not normalized.strip():
        return []

    units: list[_Unit] = []
    for title, start, end in _sections(normalized):
        for a, b in _split_span(normalized, start, end, config):
            units.append(_Unit(a, b, title))

    # Empacota unidades consecutivas da mesma seção (sem sobreposição entre elas) e anexa seções
    # pequenas à vizinha.
    merged: list[_Unit] = []
    for unit in units:
        if merged:
            previous = merged[-1]
            same_section = previous.section == unit.section
            tiny = (unit.end - unit.start) < config.min_chars or (previous.end - previous.start) < config.min_chars
            contiguous = unit.start >= previous.start
            if contiguous and (same_section or tiny) and unit.end - previous.start <= config.max_chars:
                merged[-1] = _Unit(
                    previous.start, unit.end, previous.section if same_section else _merge_titles(previous, unit)
                )
                continue
        merged.append(unit)

    spans: list[TextSpan] = []
    for unit in merged:
        start, end = _trim(normalized, unit.start, unit.end)
        if start >= end:
            continue
        chunk_text = normalized[start:end]
        spans.append(
            TextSpan(
                text=chunk_text,
                char_start=start,
                char_end=end,
                section=unit.section,
                token_estimate=estimate_tokens(chunk_text, config.chars_per_token),
            )
        )
    return spans


def _merge_titles(previous: _Unit, unit: _Unit) -> str | None:
    titles = [title for title in (previous.section, unit.section) if title]
    if not titles:
        return None
    if len(titles) == 2 and (titles[1].startswith(titles[0]) or titles[0].startswith(titles[1])):
        return max(titles, key=len)
    return " / ".join(dict.fromkeys(titles))


def split_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """API antiga (caracteres): mantida para compatibilidade; delega a ``split_document``."""
    return [span.text for span in split_document(text, ChunkingConfig.from_chars(chunk_size, overlap))]
