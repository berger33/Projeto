"""Persistência do índice com manifesto (Fase 2, R-06/R-01; decisão D4).

Layout de ``RAG_INDEX_DIR`` (padrão ``.rag_index/``)::

    manifest.json   descreve o índice: modelo e dimensão dos embeddings, versão do chunking, prefixos,
                    sha256 e tamanho de cada arquivo do corpus, nº de chunks, data de criação
    vectors.npy     matriz float32 normalizada (NumpyVectorStore)
    chunks.json     chunks serializados (texto, locator, seção, posições)

No boot, ``IndexBuilder.load_or_build`` compara o manifesto gravado com o **manifesto esperado** do
estado atual (corpus + configuração). Se forem compatíveis, o índice é carregado do disco sem chamar o
provider de embeddings; caso contrário é reconstruído e regravado, e o motivo (``reason``) vai para o
log ``index.rebuilt``. Uma troca de modelo de embedding, portanto, é sempre detectada no boot — nunca
por uma consulta que falha com dimensão errada.

Reembedar só o que mudou é intencionalmente simples: quando qualquer arquivo muda, o corpus inteiro é
reembedado (a dedup e o boilerplate são globais e um arquivo alterado pode mudar chunks de outros).
Com lotes e persistência, o custo é pago uma vez por alteração, não a cada boot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chunking import CHUNKING_VERSION, ChunkingConfig
from .config import Settings
from .documents import IngestReport, load_corpus
from .domain import Chunk
from .embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from .observability import log_event
from .retrieval import VectorIndex
from .store import NumpyVectorStore

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    manifest_version: int
    rag_mode: str
    embedding_model: str
    dimension: int
    chunking_version: str
    chunking: dict[str, Any]
    query_prefix: str
    document_prefix: str
    files: dict[str, dict[str, Any]]  # nome -> {"sha256": ..., "bytes": ...}
    chunks: int
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="seconds"))

    def compatibility_issues(self, expected: Manifest) -> list[str]:
        """Diferenças que exigem reconstrução (ignora ``created_at``, ``dimension`` e ``chunks`` do lado esperado)."""
        issues: list[str] = []
        if self.manifest_version != expected.manifest_version:
            issues.append(f"manifest_version {self.manifest_version} != {expected.manifest_version}")
        if self.rag_mode != expected.rag_mode:
            issues.append(f"rag_mode {self.rag_mode!r} != {expected.rag_mode!r}")
        if self.embedding_model != expected.embedding_model:
            issues.append(f"embedding_model {self.embedding_model!r} != {expected.embedding_model!r}")
        if self.chunking_version != expected.chunking_version or self.chunking != expected.chunking:
            issues.append("configuração de chunking mudou")
        if (self.query_prefix, self.document_prefix) != (expected.query_prefix, expected.document_prefix):
            issues.append("prefixos de tarefa mudaram")
        if self.files != expected.files:
            added = sorted(set(expected.files) - set(self.files))
            removed = sorted(set(self.files) - set(expected.files))
            changed = sorted(
                name for name in set(self.files) & set(expected.files) if self.files[name] != expected.files[name]
            )
            if added:
                issues.append(f"arquivos novos: {added}")
            if removed:
                issues.append(f"arquivos removidos: {removed}")
            if changed:
                issues.append(f"arquivos alterados: {changed}")
        return issues

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        data = json.loads(text)
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def _prefixes(embeddings: EmbeddingProvider) -> tuple[str, str]:
    if isinstance(embeddings, OllamaEmbeddingProvider):
        return embeddings.prefixes.query, embeddings.prefixes.document
    return "", ""


def expected_manifest(
    settings: Settings,
    report: IngestReport,
    embeddings: EmbeddingProvider,
    config: ChunkingConfig,
    *,
    chunks: int,
    dimension: int,
) -> Manifest:
    query_prefix, document_prefix = _prefixes(embeddings)
    return Manifest(
        manifest_version=MANIFEST_VERSION,
        rag_mode=settings.rag_mode,
        embedding_model=settings.embedding_model if settings.rag_mode == "ollama" else "hash-local",
        dimension=dimension,
        chunking_version=CHUNKING_VERSION,
        chunking=asdict(config),
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        files={entry.name: {"sha256": entry.sha256, "bytes": entry.bytes} for entry in report.files},
        chunks=chunks,
    )


@dataclass
class BuildResult:
    index: VectorIndex
    chunks: list[Chunk]
    report: IngestReport
    manifest: Manifest
    loaded_from_disk: bool
    reason: str | None  # motivo da (re)construção, None quando carregado do disco
    duration_ms: float


class IndexBuilder:
    def __init__(
        self,
        docs_dir: Path,
        settings: Settings,
        embeddings: EmbeddingProvider,
        *,
        index_dir: Path | None,
        config: ChunkingConfig | None = None,
    ):
        self.docs_dir = Path(docs_dir)
        self.settings = settings
        self.embeddings = embeddings
        self.index_dir = index_dir
        self.config = config or ChunkingConfig()

    # ----- leitura do manifesto gravado -----

    def stored_manifest(self) -> Manifest | None:
        if self.index_dir is None:
            return None
        path = self.index_dir / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            return Manifest.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, TypeError, KeyError) as exc:
            log_event(
                logger, logging.WARNING, "index.manifest_invalid", path=str(path), error=f"{type(exc).__name__}: {exc}"
            )
            return None

    # ----- API -----

    def load_or_build(self, *, force: bool = False) -> BuildResult:
        started = time.perf_counter()
        chunks, report = load_corpus(self.docs_dir, self.config)
        expected = expected_manifest(
            self.settings, report, self.embeddings, self.config, chunks=len(chunks), dimension=0
        )
        stored = self.stored_manifest()

        reason: str | None
        if force:
            reason = "forçado"
        elif self.index_dir is None:
            reason = "persistência desabilitada"
        elif stored is None:
            reason = "sem índice persistido"
        else:
            issues = stored.compatibility_issues(expected)
            if issues:
                reason = "; ".join(issues)
            elif stored.chunks != len(chunks):
                reason = f"nº de chunks {stored.chunks} != {len(chunks)}"
            else:
                reason = None

        if reason is None and stored is not None and self.index_dir is not None:
            try:
                store = NumpyVectorStore.load(self.index_dir)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                reason = f"falha ao carregar o índice persistido ({type(exc).__name__}: {exc})"
            else:
                if store.dimension != stored.dimension or [chunk.id for chunk in store.chunks] != [
                    chunk.id for chunk in chunks
                ]:
                    reason = "índice persistido não corresponde ao manifesto/corpus"
                else:
                    index = VectorIndex.from_store(store, self.embeddings)
                    duration = round((time.perf_counter() - started) * 1000.0, 2)
                    log_event(
                        logger,
                        logging.INFO,
                        "index.loaded",
                        index_dir=str(self.index_dir),
                        chunks=len(chunks),
                        dimension=store.dimension,
                        created_at=stored.created_at,
                        duration_ms=duration,
                    )
                    return BuildResult(index, store.chunks, report, stored, True, None, duration)

        index = VectorIndex(chunks, self.embeddings)
        manifest = expected_manifest(
            self.settings, report, self.embeddings, self.config, chunks=len(chunks), dimension=index.dimension
        )
        if self.index_dir is not None:
            self._save(index, manifest)
        duration = round((time.perf_counter() - started) * 1000.0, 2)
        log_event(
            logger,
            logging.INFO,
            "index.rebuilt",
            reason=reason,
            index_dir=str(self.index_dir) if self.index_dir else None,
            chunks=len(chunks),
            dimension=index.dimension,
            duration_ms=duration,
        )
        return BuildResult(index, chunks, report, manifest, False, reason, duration)

    def _save(self, index: VectorIndex, manifest: Manifest) -> None:
        store = index.store
        if self.index_dir is None or not isinstance(store, NumpyVectorStore):
            return
        try:
            store.save(self.index_dir)
            (self.index_dir / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
        except OSError as exc:
            # Persistência é otimização: falha em gravar não derruba o serviço.
            log_event(
                logger,
                logging.WARNING,
                "index.save_failed",
                index_dir=str(self.index_dir),
                error=f"{type(exc).__name__}: {exc}",
            )
