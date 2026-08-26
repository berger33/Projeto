from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    rag_mode: str = "local"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "qwen3:0.6b"
    retrieval_k: int = 5
    min_score: float = 0.12

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("RAG_MODE", "local").strip().lower()
        if mode not in {"local", "ollama"}:
            raise ValueError("RAG_MODE deve ser 'local' ou 'ollama'.")
        return cls(
            rag_mode=mode,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            generation_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6b"),
            retrieval_k=max(1, int(os.getenv("RAG_TOP_K", "5"))),
            min_score=float(os.getenv("RAG_MIN_SCORE", "0.12")),
        )
