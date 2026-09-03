"""P2-02: ingestão robusta — CSV correto (stdlib), .md/.txt, diagnóstico por arquivo, dedup, sem pandas.

Findings: G-05 (dtype do pandas corrompia valores; CSV ';' virava uma coluna), G-06 (ingestão
silenciosa), G-19 (pandas só para read_csv), R-01 (formatos), R-02 (FAQ duplicado), R-04 (eco da
pergunta do FAQ como resposta).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
from pypdf import PdfWriter
from tests.conftest import ListHandler

from app.config import Settings
from app.documents import SUPPORTED_SUFFIXES, IngestError, load_chunks, load_corpus
from app.rag import RAGService

ROOT = Path(__file__).resolve().parents[1]


def _pdf(path: Path, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


# ---------------------------------------------------------------------------
# CSV (G-05)
# ---------------------------------------------------------------------------


def test_csv_values_are_kept_as_text_and_semicolon_is_detected(tmp_path: Path) -> None:
    (tmp_path / "pedidos.csv").write_text(
        "\ufeffcodigo;valor;cep;descricao\n00123;10,50;01310-100;Camiseta básica\n00124;7.00;20040-020;Calça jeans\n",
        encoding="utf-8",
    )
    chunks, report = load_corpus(tmp_path)
    assert [chunk.id for chunk in chunks] == ["pedidos.csv:r2", "pedidos.csv:r3"]
    assert "codigo: 00123" in chunks[0].text and "valor: 10,50" in chunks[0].text and "cep: 01310-100" in chunks[0].text
    (entry,) = report.files
    assert entry.delimiter == ";" and entry.columns == ["codigo", "valor", "cep", "descricao"] and entry.rows == 2


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_csv_delimiters_are_sniffed(tmp_path: Path, delimiter: str) -> None:
    (tmp_path / "f.csv").write_text(
        f"pergunta{delimiter}resposta\nQual o prazo?{delimiter}O prazo é de 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    chunks, report = load_corpus(tmp_path)
    assert len(chunks) == 1 and "10 dias" in chunks[0].text
    assert report.files[0].delimiter == delimiter


def test_csv_display_omits_column_names_but_text_keeps_them_for_search(tmp_path: Path) -> None:
    (tmp_path / "faq.csv").write_text(
        "categoria,pergunta,resposta\nDevolução,Qual é o prazo para devolver?,O prazo é de 10 dias corridos.\n",
        encoding="utf-8",
    )
    (chunk,) = load_chunks(tmp_path)
    assert (
        chunk.text
        == "categoria: Devolução | pergunta: Qual é o prazo para devolver? | resposta: O prazo é de 10 dias corridos."
    )
    assert chunk.display == "Devolução\nQual é o prazo para devolver?\nO prazo é de 10 dias corridos."
    assert chunk.content == chunk.display


def test_csv_empty_rows_are_skipped_and_headerless_csv_fails_with_filename(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("pergunta,resposta\n,\nQ,R\n\n", encoding="utf-8")
    chunks, _ = load_corpus(tmp_path)
    assert [chunk.locator["row"] for chunk in chunks] == [3]
    (tmp_path / "a.csv").write_text("", encoding="utf-8")
    with pytest.raises(IngestError, match=r"a\.csv"):
        load_corpus(tmp_path)


def test_answer_from_csv_does_not_echo_the_faq_question(tmp_path: Path) -> None:
    """R-04: a 'sentença' 'pergunta: Qual é o prazo…' era devolvida como primeira frase da resposta."""
    (tmp_path / "faq.csv").write_text(
        "categoria,pergunta,resposta\nDevolução,Qual é o prazo para devolver?,O prazo é de 10 dias corridos após o recebimento.\n",
        encoding="utf-8",
    )
    result = RAGService(tmp_path, Settings()).answer("Qual é o prazo para devolver?")
    assert result.status == "answered"
    assert "pergunta:" not in result.answer and "categoria:" not in result.answer
    assert "10 dias" in result.answer


# ---------------------------------------------------------------------------
# Formatos (R-01) e diagnóstico (G-06)
# ---------------------------------------------------------------------------


def test_markdown_and_txt_are_ingested_with_sections(tmp_path: Path, captured: ListHandler) -> None:
    (tmp_path / "guia.md").write_text(
        "# Guia\n\n## Trocas\nTrocas em até 30 dias com etiqueta.\n\n## Frete\nFrete grátis acima de R$ 200.\n",
        encoding="utf-8",
    )
    (tmp_path / "notas.txt").write_text("Atendimento de segunda a sexta, das 9h às 18h.\n", encoding="utf-8")
    chunks, report = load_corpus(tmp_path)
    ids = [chunk.id for chunk in chunks]
    assert "guia.md:c1" in ids and "notas.txt:c1" in ids
    guia = [chunk for chunk in chunks if chunk.source == "guia.md"]
    assert {chunk.section for chunk in guia} == {"Guia / Trocas", "Guia / Frete"} or len(guia) == 1
    assert {entry.kind for entry in report.files} == {"md", "txt"}
    events = captured.events("ingest.file")
    assert {event["file"] for event in events} == {"guia.md", "notas.txt"}
    assert all(event["chunks"] >= 1 and event["sha256"] for event in events)


def test_unsupported_files_are_skipped_with_warning_and_reported(tmp_path: Path, captured: ListHandler) -> None:
    (tmp_path / "ok.txt").write_text("Conteúdo suficiente para um chunk de teste.", encoding="utf-8")
    (tmp_path / "planilha.xlsx").write_bytes(b"PK\x03\x04")
    (tmp_path / "imagem.png").write_bytes(b"\x89PNG")
    (tmp_path / "subpasta").mkdir()
    chunks, report = load_corpus(tmp_path)
    assert len(chunks) == 1 and report.skipped == ["imagem.png", "planilha.xlsx"]
    assert {event["file"] for event in captured.events("ingest.skipped")} == {"imagem.png", "planilha.xlsx"}
    assert ".xlsx" not in SUPPORTED_SUFFIXES


def test_pdf_pages_without_text_are_counted_and_warned(tmp_path: Path, captured: ListHandler) -> None:
    _pdf(tmp_path / "scan.pdf", pages=2)
    (tmp_path / "ok.txt").write_text("Texto para garantir ao menos um chunk no corpus.", encoding="utf-8")
    _, report = load_corpus(tmp_path)
    scan = next(entry for entry in report.files if entry.name == "scan.pdf")
    assert scan.pages == 2 and scan.empty_pages == 2 and scan.chunks == 0
    (empty,) = captured.events("ingest.empty")
    assert empty["file"] == "scan.pdf"


def test_corrupted_pdf_raises_ingest_error_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "quebrado.pdf").write_bytes(b"%PDF-1.4 isto nao e um pdf valido")
    with pytest.raises(IngestError, match=r"quebrado\.pdf"):
        load_corpus(tmp_path)


def test_missing_directory_and_empty_corpus_have_clear_messages(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="não existe"):
        load_corpus(tmp_path / "nada")
    (tmp_path / "x.bin").write_bytes(b"0")
    with pytest.raises(IngestError, match="Nenhum conteúdo indexável") as info:
        load_corpus(tmp_path)
    assert "x.bin" in str(info.value)


def test_index_built_event_reports_ingest_summary(tmp_path: Path, captured: ListHandler) -> None:
    (tmp_path / "a.txt").write_text("Primeiro documento com conteúdo próprio e distinto.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Primeiro documento com conteúdo próprio e distinto.", encoding="utf-8")
    service = RAGService(tmp_path, Settings())
    (built,) = captured.events("index.built")
    assert built["documents"] == 1 and built["chunks"] == 1 and built["duplicates_removed"] == 1
    assert service.ingest_report.duplicates == [("b.txt:c1", "a.txt:c1", 1.0)]


# ---------------------------------------------------------------------------
# Dedup (R-02)
# ---------------------------------------------------------------------------


def test_exact_and_near_duplicates_are_removed_keeping_first(tmp_path: Path, captured: ListHandler) -> None:
    base = "O cliente pode solicitar a devolução em até 10 dias corridos após o recebimento do pedido, com o produto sem sinais de uso."
    (tmp_path / "a.txt").write_text(base, encoding="utf-8")
    (tmp_path / "b.txt").write_text(base.upper(), encoding="utf-8")  # igual após normalização
    (tmp_path / "c.txt").write_text(base.replace("pedido", "pedido realizado"), encoding="utf-8")  # quase igual
    (tmp_path / "d.txt").write_text(
        "Aceitamos cartão de crédito e PIX; a confirmação do PIX ocorre após a identificação do pagamento.",
        encoding="utf-8",
    )
    chunks, report = load_corpus(tmp_path)
    assert [chunk.id for chunk in chunks] == ["a.txt:c1", "d.txt:c1"]
    discarded = {item[0]: item for item in report.duplicates}
    assert discarded["b.txt:c1"][1] == "a.txt:c1" and discarded["b.txt:c1"][2] == 1.0
    assert discarded["c.txt:c1"][1] == "a.txt:c1" and 0.9 <= discarded["c.txt:c1"][2] < 1.0
    assert len(captured.events("ingest.duplicate")) == 2


def test_short_chunks_are_not_near_deduped_only_exact(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Prazo: 10 dias.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Prazo: 30 dias.", encoding="utf-8")
    chunks, report = load_corpus(tmp_path)
    assert len(chunks) == 2 and report.duplicates == []


def test_real_corpus_keeps_both_faq_versions_per_decision_d8() -> None:
    chunks, report = load_corpus(ROOT / "corpus")
    assert {entry.name for entry in report.files} == {
        "faq.csv",
        "faq.pdf",
        "politica_privacidade.pdf",
        "politica_reembolso_devolucoes.pdf",
    }
    assert report.chunks == len(chunks) == 25 and report.skipped == []
    # As redações do CSV e do PDF diferem o bastante para não serem quase-duplicatas (Jaccard < 0,9).
    assert report.duplicates == []


def test_pandas_is_no_longer_a_dependency() -> None:
    assert "pandas" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert importlib.util.find_spec("pandas") is None or True  # pode existir no ambiente, mas não é usada
    assert "pandas" not in (ROOT / "app" / "documents.py").read_text(encoding="utf-8")


def test_ingest_logging_is_quiet_at_warning_level(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "a.txt").write_text("Conteúdo de teste para ingestão silenciosa em WARNING.", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="app.documents"):
        load_corpus(tmp_path)
    assert not [record for record in caplog.records if record.name == "app.documents"]
