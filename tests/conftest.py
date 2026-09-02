"""Fixtures compartilhadas entre os módulos de teste."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

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
