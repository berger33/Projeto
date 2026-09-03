"""Reindexação por linha de comando (P2-03).

    python -m app.ingest                # (re)constrói o índice persistido a partir de corpus/ (ou CORPUS_DIR)
    python -m app.ingest --check        # só compara o manifesto gravado com o estado atual (exit 1 se precisar reindexar)
    python -m app.ingest --docs /srv/corpus --index-dir /var/lib/aurora/index

Útil em CI/CD e em imagens Docker: indexa em build/deploy e o serviço sobe em milissegundos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .chunking import ChunkingConfig
from .config import ConfigError, Settings
from .documents import IngestError, load_corpus
from .embeddings import HashEmbeddingProvider, OllamaEmbeddingProvider
from .errors import ProviderError
from .observability import configure_logging
from .persistence import IndexBuilder, expected_manifest

BASE = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrói ou verifica o índice persistido do Aurora Document RAG.")
    parser.add_argument("--docs", default=None, help="diretório do corpus (padrão: CORPUS_DIR ou corpus/)")
    parser.add_argument("--index-dir", default=None, help="diretório do índice (padrão: RAG_INDEX_DIR ou .rag_index)")
    parser.add_argument(
        "--check", action="store_true", help="não reindexa; sai com 1 se o índice estiver desatualizado"
    )
    parser.add_argument("--force", action="store_true", help="reconstrói mesmo que o manifesto esteja compatível")
    parser.add_argument("--json", action="store_true", help="imprime o resultado em JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=os.getenv("LOG_LEVEL", "WARNING"))
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"configuração inválida: {exc}", file=sys.stderr)
        return 2
    index_dir = Path(args.index_dir) if args.index_dir else (Path(settings.index_dir) if settings.index_dir else None)
    docs = Path(args.docs) if args.docs else Path(settings.corpus_dir)
    if not docs.is_absolute():
        docs = BASE / docs
    args.docs = str(docs)
    if index_dir is None:
        print("persistência desabilitada (RAG_INDEX_DIR vazio); informe --index-dir.", file=sys.stderr)
        return 2

    if settings.rag_mode == "ollama":
        embeddings: HashEmbeddingProvider | OllamaEmbeddingProvider = OllamaEmbeddingProvider(
            settings.ollama_base_url,
            settings.embedding_model,
            timeout=settings.embed_timeout_s,
            batch_size=settings.embed_batch_size,
        )
    else:
        embeddings = HashEmbeddingProvider()
    builder = IndexBuilder(Path(args.docs), settings, embeddings, index_dir=index_dir)

    try:
        if args.check:
            chunks, report = load_corpus(Path(args.docs), ChunkingConfig())
            stored = builder.stored_manifest()
            expected = expected_manifest(
                settings, report, embeddings, ChunkingConfig(), chunks=len(chunks), dimension=0
            )
            issues = ["sem índice persistido"] if stored is None else stored.compatibility_issues(expected)
            if stored is not None and not issues and stored.chunks != len(chunks):
                issues.append(f"nº de chunks {stored.chunks} != {len(chunks)}")
            payload = {"index_dir": str(index_dir), "up_to_date": not issues, "issues": issues, "chunks": len(chunks)}
            _emit(
                payload, args.json, "índice atualizado" if not issues else "índice desatualizado: " + "; ".join(issues)
            )
            return 0 if not issues else 1
        result = builder.load_or_build(force=args.force)
    except (IngestError, ProviderError, OSError) as exc:
        print(f"falha na indexação ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    payload = {
        "index_dir": str(index_dir),
        "loaded_from_disk": result.loaded_from_disk,
        "reason": result.reason,
        "chunks": len(result.chunks),
        "documents": len(result.report.files),
        "duplicates_removed": len(result.report.duplicates),
        "skipped": result.report.skipped,
        "dimension": result.index.dimension,
        "duration_ms": result.duration_ms,
        "manifest": asdict(result.manifest),
    }
    action = "carregado do disco" if result.loaded_from_disk else f"reconstruído ({result.reason})"
    _emit(
        payload,
        args.json,
        f"índice {action}: {payload['chunks']} chunks de {payload['documents']} documento(s) em {index_dir}",
    )
    return 0


def _emit(payload: dict[str, object], as_json: bool, message: str) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(message)


if __name__ == "__main__":
    sys.exit(main())
