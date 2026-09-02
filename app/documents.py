from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .chunking import ChunkingConfig, detect_boilerplate, split_document, split_text
from .domain import Chunk

__all__ = ["ChunkingConfig", "load_chunks", "split_text"]


def _compact(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def load_chunks(docs_dir: str | Path, config: ChunkingConfig | None = None) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    paths = sorted(docs_dir.glob("*"))

    # Cabeçalho/rodapé repetidos são detectados sobre todas as páginas de todos os PDFs do corpus
    # (papel timbrado se repete entre documentos) e removidos antes do chunking (R-02).
    pdf_pages: dict[Path, list[str]] = {}
    for path in paths:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            pdf_pages[path] = [page.extract_text() or "" for page in reader.pages]
    boilerplate = detect_boilerplate(text for pages in pdf_pages.values() for text in pages)

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page_number, text in enumerate(pdf_pages[path], start=1):
                for position, span in enumerate(split_document(text, config, boilerplate=boilerplate), start=1):
                    chunks.append(
                        Chunk(
                            id=f"{path.name}:p{page_number}:c{position}",
                            text=span.text,
                            source=path.name,
                            locator={"page": page_number},
                            section=span.section,
                            char_start=span.char_start,
                            char_end=span.char_end,
                            token_estimate=span.token_estimate,
                        )
                    )
        elif suffix == ".csv":
            frame = pd.read_csv(path).fillna("")
            for row_index, row in frame.iterrows():
                text = " | ".join(f"{column}: {row[column]}" for column in frame.columns)
                chunks.append(
                    Chunk(
                        id=f"{path.name}:r{int(row_index) + 2}",
                        text=_compact(text),
                        source=path.name,
                        locator={"row": int(row_index) + 2},
                    )
                )
    if not chunks:
        raise RuntimeError(f"Nenhum conteúdo PDF/CSV encontrado em {docs_dir}.")
    return chunks
