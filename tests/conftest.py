"""Fixtures compartilhadas entre os módulos de teste."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.observability import JsonFormatter


class ListHandler(logging.Handler):
    """Captura registros já formatados como JSON (exercita o formatter real)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(JsonFormatter())
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(self.format(record)))

    def events(self, name: str) -> list[dict]:
        return [record for record in self.records if record["event"] == name]


@pytest.fixture()
def captured() -> Iterator[ListHandler]:
    """Handler no logger raiz em DEBUG; devolve os eventos emitidos durante o teste."""
    root = logging.getLogger()
    handler = ListHandler()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


@pytest.fixture(autouse=True)
def _isolated_index_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cada teste persiste o índice num diretório temporário (nunca no ``.rag_index`` do repositório)."""
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path_factory.mktemp("rag_index")))


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Gera um PDF real com texto extraível (fonte Helvetica embutida no leitor), uma lista de linhas por página."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        body = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in escaped) + " ET"
        stream = DecodedStreamObject()
        stream.set_data(body.encode("cp1252"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)
    return path
