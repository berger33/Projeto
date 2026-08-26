from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .domain import Chunk


def _compact(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def split_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = _compact(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            buffer = f"{tail}\n\n{paragraph}".strip()
        else:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + chunk_size)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk]


def load_chunks(docs_dir: str | Path) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                for position, part in enumerate(split_text(page.extract_text() or ""), start=1):
                    chunks.append(Chunk(id=f"{path.name}:p{page_number}:c{position}", text=part, source=path.name, locator={"page": page_number}))
        elif suffix == ".csv":
            frame = pd.read_csv(path).fillna("")
            for row_index, row in frame.iterrows():
                text = " | ".join(f"{column}: {row[column]}" for column in frame.columns)
                chunks.append(Chunk(id=f"{path.name}:r{int(row_index)+2}", text=_compact(text), source=path.name, locator={"row": int(row_index)+2}))
    if not chunks:
        raise RuntimeError(f"Nenhum conteúdo PDF/CSV encontrado em {docs_dir}.")
    return chunks
