from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Protocol

import httpx

from .observability import log_event, ns_to_ms

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+", re.UNICODE)


class HashEmbeddingProvider:
    """Embedding determinístico para CI/offline; não pretende ser semântico."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

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


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
            response.raise_for_status()
            payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Resposta inválida do endpoint de embeddings do Ollama.")
        log_event(
            logger,
            logging.INFO,
            "provider.embed",
            model=self.model,
            texts=len(texts),
            chars=sum(len(text) for text in texts),
            dimension=len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None,
            prompt_tokens=payload.get("prompt_eval_count"),
            ollama_total_ms=ns_to_ms(payload.get("total_duration")),
            ollama_load_ms=ns_to_ms(payload.get("load_duration")),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return embeddings
