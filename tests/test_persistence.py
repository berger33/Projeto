"""P2-03: manifesto de ingestão + persistência do índice + reload (R-06, R-01 versionamento/atualização)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.conftest import ListHandler

from app import ingest as ingest_cli
from app.chunking import CHUNKING_VERSION, ChunkingConfig
from app.config import Settings
from app.embeddings import HashEmbeddingProvider, OllamaEmbeddingProvider
from app.persistence import MANIFEST_FILE, IndexBuilder, Manifest
from app.rag import RAGService


class CountingHash(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "faq.csv").write_text(
        "pergunta,resposta\nprazo,O prazo de devolução é de 10 dias corridos após o recebimento.\npagamento,Aceitamos PIX e cartão.\n",
        encoding="utf-8",
    )
    (docs / "guia.txt").write_text(
        "O suporte responde pelo e-mail suporte@auroramoda.exemplo em até dois dias úteis.", encoding="utf-8"
    )
    return docs


def _builder(
    corpus: Path, index_dir: Path | None, embeddings: HashEmbeddingProvider | None = None, **settings: Any
) -> IndexBuilder:
    return IndexBuilder(corpus, Settings(**settings), embeddings or CountingHash(), index_dir=index_dir)


# ---------------------------------------------------------------------------
# Manifesto e ciclo build → load
# ---------------------------------------------------------------------------


def test_first_boot_builds_and_persists_then_second_boot_loads_without_embedding(
    corpus: Path, tmp_path: Path, captured: ListHandler
) -> None:
    index_dir = tmp_path / "idx"
    embeddings = CountingHash()
    first = _builder(corpus, index_dir, embeddings).load_or_build()
    assert not first.loaded_from_disk and first.reason == "sem índice persistido"
    assert embeddings.document_calls == 1
    assert {path.name for path in index_dir.iterdir()} == {MANIFEST_FILE, "vectors.npy", "chunks.json"}
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "hash-local" and manifest["dimension"] == 384
    assert manifest["chunking_version"] == CHUNKING_VERSION and set(manifest["files"]) == {"faq.csv", "guia.txt"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"].values())

    second = _builder(corpus, index_dir, embeddings).load_or_build()
    assert second.loaded_from_disk and second.reason is None
    assert embeddings.document_calls == 1  # nada reembedado
    assert [chunk.id for chunk in second.chunks] == [chunk.id for chunk in first.chunks]
    assert second.index.scores("prazo") == pytest.approx(first.index.scores("prazo"), abs=1e-6)
    assert captured.events("index.loaded") and captured.events("index.rebuilt")


def test_changed_file_triggers_rebuild_with_reason(corpus: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "idx"
    _builder(corpus, index_dir).load_or_build()
    (corpus / "guia.txt").write_text("Texto novo: o suporte atende também por chat das 9h às 18h.", encoding="utf-8")
    result = _builder(corpus, index_dir).load_or_build()
    assert not result.loaded_from_disk and "arquivos alterados: ['guia.txt']" in (result.reason or "")
    assert any("chat" in chunk.text for chunk in result.chunks)


def test_added_and_removed_files_are_reported(corpus: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "idx"
    _builder(corpus, index_dir).load_or_build()
    (corpus / "novo.md").write_text("# Trocas\nTrocas em até 30 dias.", encoding="utf-8")
    result = _builder(corpus, index_dir).load_or_build()
    assert "arquivos novos: ['novo.md']" in (result.reason or "")
    (corpus / "novo.md").unlink()
    (corpus / "guia.txt").unlink()
    result = _builder(corpus, index_dir).load_or_build()
    assert "arquivos removidos: ['guia.txt', 'novo.md']" in (result.reason or "")


def test_embedding_model_or_prefix_change_is_detected_at_boot(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = tmp_path / "idx"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    settings = Settings(rag_mode="ollama", embedding_model="nomic-embed-text-v2-moe")
    IndexBuilder(
        corpus, settings, OllamaEmbeddingProvider("http://o:1", settings.embedding_model), index_dir=index_dir
    ).load_or_build()

    other = Settings(rag_mode="ollama", embedding_model="embeddinggemma")
    result = IndexBuilder(
        corpus, other, OllamaEmbeddingProvider("http://o:1", other.embedding_model), index_dir=index_dir
    ).load_or_build()
    assert not result.loaded_from_disk
    assert "embedding_model 'nomic-embed-text-v2-moe' != 'embeddinggemma'" in (result.reason or "")
    assert "prefixos de tarefa mudaram" in (result.reason or "")

    # Voltar para o modo local com o mesmo diretório também reconstrói (rag_mode + modelo diferentes).
    result = _builder(corpus, index_dir).load_or_build()
    assert "rag_mode 'ollama' != 'local'" in (result.reason or "")


def test_chunking_config_change_and_force_trigger_rebuild(corpus: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "idx"
    _builder(corpus, index_dir).load_or_build()
    other = IndexBuilder(corpus, Settings(), CountingHash(), index_dir=index_dir, config=ChunkingConfig(max_tokens=64))
    assert "chunking" in (other.load_or_build().reason or "")
    forced = _builder(corpus, index_dir).load_or_build(force=True)
    assert forced.reason == "forçado"


def test_corrupted_manifest_or_vectors_fall_back_to_rebuild(
    corpus: Path, tmp_path: Path, captured: ListHandler
) -> None:
    index_dir = tmp_path / "idx"
    _builder(corpus, index_dir).load_or_build()
    (index_dir / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    result = _builder(corpus, index_dir).load_or_build()
    assert result.reason == "sem índice persistido" and captured.events("index.manifest_invalid")

    (index_dir / "vectors.npy").write_bytes(b"garbage")
    result = _builder(corpus, index_dir).load_or_build()
    assert not result.loaded_from_disk and "falha ao carregar" in (result.reason or "")
    # Após a reconstrução, o índice volta a ser carregável.
    assert _builder(corpus, index_dir).load_or_build().loaded_from_disk


def test_persistence_disabled_rebuilds_every_boot(corpus: Path) -> None:
    result = _builder(corpus, None).load_or_build()
    assert not result.loaded_from_disk and result.reason == "persistência desabilitada"
    assert _builder(corpus, None).stored_manifest() is None


def test_manifest_round_trip_and_compatibility_ignores_timestamp_and_dimension() -> None:
    base = Manifest(
        1, "local", "hash-local", 384, "2", {"max_tokens": 300}, "", "", {"a.csv": {"sha256": "x", "bytes": 1}}, 3
    )
    clone = Manifest.from_json(base.to_json())
    assert clone == base
    later = Manifest(
        1,
        "local",
        "hash-local",
        0,
        "2",
        {"max_tokens": 300},
        "",
        "",
        {"a.csv": {"sha256": "x", "bytes": 1}},
        3,
        created_at="2030-01-01T00:00:00+00:00",
    )
    assert base.compatibility_issues(later) == []
    assert (
        Manifest.from_json(
            '{"manifest_version": 1, "rag_mode": "local", "embedding_model": "m", "dimension": 1, "chunking_version": "2", "chunking": {}, "query_prefix": "", "document_prefix": "", "files": {}, "chunks": 0, "extra": true}'
        ).chunks
        == 0
    )


# ---------------------------------------------------------------------------
# RAGService: boot rápido com índice persistido; reload()
# ---------------------------------------------------------------------------


def test_service_boots_from_persisted_index_and_logs_it(corpus: Path, tmp_path: Path, captured: ListHandler) -> None:
    index_dir = tmp_path / "idx"
    RAGService(corpus, Settings(), index_dir=index_dir)
    RAGService(corpus, Settings(), index_dir=index_dir)
    built = captured.events("index.built")
    assert [event["loaded_from_disk"] for event in built] == [False, True]
    assert built[0]["rebuild_reason"] == "sem índice persistido" and built[1]["rebuild_reason"] is None
    assert built[1]["index_dir"] == str(index_dir)


def test_service_reload_picks_up_new_documents_without_restart(
    corpus: Path, tmp_path: Path, captured: ListHandler
) -> None:
    service = RAGService(corpus, Settings(), index_dir=tmp_path / "idx")
    assert service.answer("Posso trocar o produto por outro tamanho?").status == "refused_no_context"
    (corpus / "trocas.txt").write_text(
        "Trocas: o cliente pode trocar o produto por outro tamanho em até 30 dias corridos após o recebimento, com etiqueta.",
        encoding="utf-8",
    )
    summary = service.reload()
    assert summary["documents"] == 3 and summary["chunks"] == service.chunk_count
    result = service.answer("Posso trocar o produto por outro tamanho?")
    assert result.status == "answered" and any(source.document == "trocas.txt" for source in result.sources)
    (event,) = captured.events("index.reloaded")
    assert event["summary"]["chunks"] == service.chunk_count
    assert service.manifest.files.keys() == {"faq.csv", "guia.txt", "trocas.txt"}


def test_settings_index_dir_env_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings.from_env({}).index_dir == ".rag_index"
    assert Settings.from_env({"RAG_INDEX_DIR": " /var/idx "}).index_dir == "/var/idx"
    assert Settings.from_env({"RAG_INDEX_DIR": ""}).index_dir == ""
    assert Settings.from_env({"RAG_INDEX_DIR": ""}).public_dict()["index_dir"] is None


# ---------------------------------------------------------------------------
# CLI python -m app.ingest
# ---------------------------------------------------------------------------


def test_ingest_cli_builds_checks_and_reports(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = tmp_path / "cli_idx"
    monkeypatch.setenv("RAG_MODE", "local")
    assert ingest_cli.main(["--docs", str(corpus), "--index-dir", str(index_dir), "--check"]) == 1
    assert "desatualizado" in capsys.readouterr().out

    assert ingest_cli.main(["--docs", str(corpus), "--index-dir", str(index_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["loaded_from_disk"] is False and payload["chunks"] == 3 and payload["manifest"]["chunks"] == 3

    assert ingest_cli.main(["--docs", str(corpus), "--index-dir", str(index_dir), "--check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["up_to_date"] is True

    assert ingest_cli.main(["--docs", str(corpus), "--index-dir", str(index_dir)]) == 0
    assert "carregado do disco" in capsys.readouterr().out
    assert ingest_cli.main(["--docs", str(corpus), "--index-dir", str(index_dir), "--force"]) == 0
    assert "reconstruído (forçado)" in capsys.readouterr().out


def test_ingest_cli_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_MODE", "local")
    assert ingest_cli.main(["--docs", str(tmp_path / "nada"), "--index-dir", str(tmp_path / "i")]) == 1
    assert "falha na indexação" in capsys.readouterr().err
    monkeypatch.setenv("RAG_INDEX_DIR", "")
    assert ingest_cli.main(["--docs", str(tmp_path)]) == 2
    assert "persistência desabilitada" in capsys.readouterr().err
    monkeypatch.setenv("RAG_TOP_K", "zero")
    assert ingest_cli.main([]) == 2
