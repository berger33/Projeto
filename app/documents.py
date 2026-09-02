"""Ingestão do corpus: PDF, CSV, Markdown e texto → ``Chunk`` rastreáveis (Fase 2, G-05/G-06/R-01/R-02/R-04).

- **CSV** com ``csv`` da biblioteca padrão: delimitador detectado (``,`` ``;`` ``\\t`` ``|``), BOM tolerado
  (``utf-8-sig``), tudo tratado como texto (``00123`` continua ``00123``). Cada linha vira um chunk cujo
  ``text`` (o que é embedado/buscado) reúne todas as colunas, e cuja ``display`` (o que é mostrado ao
  gerador e ao usuário) é só o conteúdo — sem ``coluna:`` — para que a pergunta do FAQ não seja ecoada
  como resposta.
- **PDF** via ``pypdf``: uma página por vez, chunking por seção (``app.chunking``) com boilerplate
  detectado sobre todas as páginas do corpus.
- **Markdown/texto** (``.md``, ``.txt``): o arquivo inteiro passa pelo chunking (títulos ``#`` viram seções).
- **Diagnóstico por arquivo** (evento ``ingest.file``): formato, páginas, páginas sem texto, linhas,
  chunks; arquivos não suportados geram ``ingest.skipped``; falha de leitura vira ``IngestError`` com o
  nome do arquivo.
- **Dedup**: chunks com texto normalizado idêntico são descartados (fica o primeiro na ordem dos
  arquivos); quase-duplicatas (Jaccard de radicais >= 0,9) também, com log ``ingest.duplicate``.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .chunking import ChunkingConfig, detect_boilerplate, split_document, split_text
from .domain import Chunk
from .lexical import analyze
from .observability import log_event
from .text import normalize

__all__ = ["ChunkingConfig", "IngestError", "IngestReport", "load_chunks", "load_corpus", "split_text"]

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".pdf", ".csv", ".md", ".txt", ".markdown"})
NEAR_DUPLICATE_JACCARD = 0.9
_CSV_DELIMITERS = ",;\t|"


class IngestError(RuntimeError):
    """Falha ao ler um arquivo do corpus; a mensagem cita o arquivo."""


@dataclass
class FileReport:
    name: str
    kind: str
    chunks: int = 0
    pages: int | None = None
    empty_pages: int = 0
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    delimiter: str | None = None
    sha256: str = ""
    bytes: int = 0
    duration_ms: float = 0.0


@dataclass
class IngestReport:
    directory: str
    files: list[FileReport] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str, float]] = field(default_factory=list)  # (descartado, mantido, similaridade)
    chunks: int = 0

    @property
    def documents(self) -> int:
        return len(self.files)


# ---------------------------------------------------------------------------
# Leitores
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    """``(colunas, linhas, delimitador)`` — tudo como texto, delimitador detectado, BOM tolerado."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")
    sample = raw[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
    columns = [column.strip() for column in (reader.fieldnames or []) if column and column.strip()]
    if not columns:
        raise IngestError(f"{path.name}: CSV sem cabeçalho ou vazio")
    rows: list[dict[str, str]] = []
    for row in reader:
        # ``line_num`` é a linha física do arquivo (cabeçalho = 1): preservada em ``_row`` para o locator,
        # mesmo quando linhas vazias no meio do arquivo são descartadas.
        cleaned = {
            str(key).strip(): (value if isinstance(value, str) else " ".join(value or [])).strip()
            for key, value in row.items()
            if key is not None
        }
        if any(cleaned.values()):
            cleaned["_row"] = str(reader.line_num)
            rows.append(cleaned)
    return columns, rows, delimiter


def _csv_chunks(path: Path, columns: list[str], rows: list[dict[str, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for row in rows:
        offset = int(row["_row"])
        values = [(column, row.get(column, "")) for column in columns]
        text = " | ".join(f"{column}: {value}" for column, value in values if value)
        display = "\n".join(value for _, value in values if value)
        if not text:
            continue
        chunks.append(
            Chunk(
                id=f"{path.name}:r{offset}",
                text=" ".join(text.split()),
                source=path.name,
                locator={"row": offset},
                display=display,
                token_estimate=None,
            )
        )
    return chunks


def _pdf_pages(path: Path) -> list[str]:
    try:
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as exc:
        raise IngestError(f"{path.name}: não foi possível ler o PDF ({type(exc).__name__}: {exc})") from exc


def _text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _spans_to_chunks(
    path: Path, text: str, config: ChunkingConfig, boilerplate: frozenset[str], *, page: int | None
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for position, span in enumerate(split_document(text, config, boilerplate=boilerplate), start=1):
        prefix = f"{path.name}:p{page}" if page is not None else path.name
        chunks.append(
            Chunk(
                id=f"{prefix}:c{position}",
                text=span.text,
                source=path.name,
                locator={"page": page} if page is not None else {},
                section=span.section,
                char_start=span.char_start,
                char_end=span.char_end,
                token_estimate=span.token_estimate,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _dedupe(chunks: list[Chunk], report: IngestReport) -> list[Chunk]:
    kept: list[Chunk] = []
    seen_exact: dict[str, Chunk] = {}
    kept_terms: list[set[str]] = []
    for chunk in chunks:
        key = normalize(chunk.display or chunk.text)
        if key in seen_exact:
            report.duplicates.append((chunk.id, seen_exact[key].id, 1.0))
            continue
        terms = set(analyze(chunk.display or chunk.text))
        duplicate_of: tuple[Chunk, float] | None = None
        if len(terms) >= 8:
            for other, other_terms in zip(kept, kept_terms, strict=True):
                if not other_terms:
                    continue
                jaccard = len(terms & other_terms) / len(terms | other_terms)
                if jaccard >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = (other, round(jaccard, 3))
                    break
        if duplicate_of is not None:
            report.duplicates.append((chunk.id, duplicate_of[0].id, duplicate_of[1]))
            continue
        seen_exact[key] = chunk
        kept.append(chunk)
        kept_terms.append(terms)
    return kept


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def load_corpus(docs_dir: str | Path, config: ChunkingConfig | None = None) -> tuple[list[Chunk], IngestReport]:
    """Lê todos os arquivos suportados de ``docs_dir`` (ordem alfabética) e devolve chunks + relatório."""
    docs_dir = Path(docs_dir)
    config = config or ChunkingConfig()
    report = IngestReport(directory=str(docs_dir))
    if not docs_dir.is_dir():
        raise IngestError(f"Diretório do corpus não existe ou não é um diretório: {docs_dir}")

    paths = sorted(path for path in docs_dir.iterdir() if path.is_file())
    pdf_pages: dict[Path, list[str]] = {}
    for path in paths:
        if path.suffix.lower() == ".pdf":
            pdf_pages[path] = _pdf_pages(path)
    boilerplate = detect_boilerplate(text for pages in pdf_pages.values() for text in pages)

    chunks: list[Chunk] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            report.skipped.append(path.name)
            log_event(logger, logging.WARNING, "ingest.skipped", file=path.name, reason="formato não suportado")
            continue
        started = time.perf_counter()
        entry = FileReport(name=path.name, kind=suffix.lstrip("."), sha256=_sha256(path), bytes=path.stat().st_size)
        produced: list[Chunk] = []
        if suffix == ".pdf":
            pages = pdf_pages[path]
            entry.pages = len(pages)
            for number, text in enumerate(pages, start=1):
                if not text.strip():
                    entry.empty_pages += 1
                    continue
                produced.extend(_spans_to_chunks(path, text, config, boilerplate, page=number))
        elif suffix == ".csv":
            columns, rows, delimiter = _read_csv(path)
            entry.rows, entry.columns, entry.delimiter = len(rows), columns, delimiter
            produced = _csv_chunks(path, columns, rows)
        else:
            produced = _spans_to_chunks(path, _text_file(path), config, boilerplate, page=None)
        entry.chunks = len(produced)
        entry.duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        report.files.append(entry)
        log_event(
            logger,
            logging.INFO,
            "ingest.file",
            file=entry.name,
            kind=entry.kind,
            bytes=entry.bytes,
            sha256=entry.sha256[:12],
            pages=entry.pages,
            empty_pages=entry.empty_pages or None,
            rows=entry.rows,
            columns=entry.columns or None,
            delimiter=entry.delimiter,
            chunks=entry.chunks,
            duration_ms=entry.duration_ms,
        )
        if entry.chunks == 0:
            log_event(logger, logging.WARNING, "ingest.empty", file=entry.name, kind=entry.kind)
        chunks.extend(produced)

    chunks = _dedupe(chunks, report)
    for discarded, kept, similarity in report.duplicates:
        log_event(logger, logging.INFO, "ingest.duplicate", discarded=discarded, kept=kept, similarity=similarity)
    report.chunks = len(chunks)
    if not chunks:
        raise IngestError(
            f"Nenhum conteúdo indexável em {docs_dir} (formatos: {', '.join(sorted(SUPPORTED_SUFFIXES))}; "
            f"arquivos ignorados: {report.skipped or 'nenhum'})."
        )
    return chunks, report


def load_chunks(docs_dir: str | Path, config: ChunkingConfig | None = None) -> list[Chunk]:
    return load_corpus(docs_dir, config)[0]
