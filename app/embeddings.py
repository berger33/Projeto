from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx


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
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
            response.raise_for_status()
            payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Resposta inválida do endpoint de embeddings do Ollama.")
        return embeddings
