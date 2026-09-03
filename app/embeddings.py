from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .errors import ProviderError, ProviderResponseError, ProviderTimeoutError, ProviderUnavailableError, provider_call
from .observability import log_event, ns_to_ms
from .ollama_client import OllamaGate, SharedClient, no_gate

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Embeddings assimétricos: documentos e consultas podem receber prefixos de tarefa distintos."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+", re.UNICODE)


class HashEmbeddingProvider:
    """Embedding determinístico para CI/offline; não pretende ser semântico."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % self.dimensions
            sign = -1.0 if (number >> 1) & 1 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# Prefixos de tarefa por família de modelo (Fase 2, R-05)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskPrefixes:
    query: str = ""
    document: str = ""


# Ordem importa: o primeiro padrão que casar com o nome do modelo vence. Fontes: model cards.
_PREFIX_TABLE: tuple[tuple[str, TaskPrefixes], ...] = (
    ("nomic-embed", TaskPrefixes(query="search_query: ", document="search_document: ")),
    ("embeddinggemma", TaskPrefixes(query="task: search result | query: ", document="title: none | text: ")),
    (
        "qwen3-embedding",
        TaskPrefixes(
            query="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        ),
    ),
    ("mxbai-embed", TaskPrefixes(query="Represent this sentence for searching relevant passages: ")),
    ("e5", TaskPrefixes(query="query: ", document="passage: ")),
    ("bge-m3", TaskPrefixes()),
    ("snowflake-arctic-embed", TaskPrefixes(query="Represent this sentence for searching relevant passages: ")),
)


def default_prefixes(model: str) -> TaskPrefixes:
    """Prefixos documentados para a família do modelo; vazio para modelos que não usam prefixo."""
    name = model.lower()
    for needle, prefixes in _PREFIX_TABLE:
        if needle in name:
            return prefixes
    return TaskPrefixes()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaEmbeddingProvider:
    """Cliente de ``/api/embed`` com prefixos de tarefa, lotes, retry e validação de dimensão.

    - Documentos e consultas recebem os prefixos da família do modelo (``prefixes``; ``None`` = padrão
      por nome; ``TaskPrefixes()`` = desliga).
    - ``embed_documents`` envia lotes de ``batch_size`` textos por requisição, reutilizando uma conexão.
    - Falhas transitórias (timeout, conexão, HTTP 5xx/429) são repetidas ``max_retries`` vezes com
      backoff exponencial; 4xx (modelo inexistente, payload inválido) falham imediatamente.
    - Todos os vetores devolvidos precisam ter a mesma dimensão; divergência é erro do provider.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        *,
        prefixes: TaskPrefixes | None = None,
        batch_size: int = 32,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        truncate: bool = True,
        gate: OllamaGate | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prefixes = default_prefixes(model) if prefixes is None else prefixes
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.backoff_s = max(0.0, backoff_s)
        self.truncate = truncate
        self.dimension: int | None = None
        self.gate = gate or no_gate()
        self._client = SharedClient(timeout)

    # ----- API pública -----

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [f"{self.prefixes.document}{text}" for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self.batch_size):
            batch = prefixed[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, kind="documents"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([f"{self.prefixes.query}{text}"], kind="query")[0]

    def close(self) -> None:
        self._client.close()

    # ----- internos -----

    def _embed_batch(self, batch: list[str], *, kind: str) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                with self.gate.acquire() as queue_wait_ms, provider_call("embed", url):
                    response = self._client.get().post(
                        url, json={"model": self.model, "input": batch, "truncate": self.truncate}
                    )
                    response.raise_for_status()
                    payload = response.json()
                break
            except ProviderError as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise
                attempt += 1
                delay = self.backoff_s * (2 ** (attempt - 1))
                log_event(
                    logger,
                    logging.WARNING,
                    "provider.retry",
                    operation="embed",
                    model=self.model,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    delay_s=delay,
                    error_code=exc.error_code,
                )
                if delay:
                    time.sleep(delay)

        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise ProviderResponseError(
                f"/api/embed devolveu {len(embeddings) if isinstance(embeddings, list) else 'nenhum'} "
                f"embedding(s) para {len(batch)} texto(s) (modelo {self.model})"
            )
        self._check_dimension(embeddings)
        log_event(
            logger,
            logging.INFO,
            "provider.embed",
            model=self.model,
            kind=kind,
            texts=len(batch),
            chars=sum(len(text) for text in batch),
            dimension=self.dimension,
            attempts=attempt + 1,
            queue_wait_ms=queue_wait_ms,
            prompt_tokens=payload.get("prompt_eval_count"),
            ollama_total_ms=ns_to_ms(payload.get("total_duration")),
            ollama_load_ms=ns_to_ms(payload.get("load_duration")),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return embeddings

    def _check_dimension(self, embeddings: list[list[float]]) -> None:
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise ProviderResponseError(f"/api/embed devolveu vetor vazio ou inválido (modelo {self.model})")
            if self.dimension is None:
                self.dimension = len(vector)
            elif len(vector) != self.dimension:
                raise ProviderResponseError(
                    f"/api/embed devolveu vetor de dimensão {len(vector)}; esperado {self.dimension} "
                    f"(modelo {self.model})"
                )


def _is_transient(exc: ProviderError) -> bool:
    if isinstance(exc, ProviderTimeoutError):
        return True
    if isinstance(exc, ProviderUnavailableError):
        cause = exc.__cause__
        if isinstance(cause, httpx.HTTPStatusError):
            return cause.response.status_code == 429 or cause.response.status_code >= 500
        return True  # falha de conexão
    return False
