"""Configuração da aplicação a partir de variáveis de ambiente, validada no boot.

Toda variável malformada ou fora de faixa gera ``ConfigError`` com o nome da variável e a faixa
aceita, de modo que a aplicação falhe ao subir (``lifespan``) em vez de na primeira requisição
(Fase 2, G-04). ``Settings`` continua um ``dataclass`` congelado: a validação também roda quando o
objeto é construído diretamente em código/testes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

RAG_MODES = ("local", "ollama")

# (mínimo, máximo) inclusivos
TOP_K_RANGE = (1, 50)
MIN_SCORE_RANGE = (0.0, 1.0)
TIMEOUT_RANGE = (1.0, 600.0)
BATCH_SIZE_RANGE = (1, 256)


class ConfigError(ValueError):
    """Configuração inválida. A mensagem é escrita para o operador e cita a variável de ambiente."""


@dataclass(frozen=True)
class Settings:
    rag_mode: str = "local"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text-v2-moe"
    generation_model: str = "qwen3:0.6b"
    retrieval_k: int = 5
    min_score: float = 0.12
    # Cosseno a partir do qual um chunk é aceito mesmo sem sobreposição lexical com a pergunta.
    vector_only_min_score: float = 0.5
    embed_timeout_s: float = 30.0
    generate_timeout_s: float = 60.0
    embed_batch_size: int = 32
    # Só informativo (não valida): de onde cada valor veio, para o log ``settings.loaded``.
    source: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.rag_mode not in RAG_MODES:
            errors.append(f"RAG_MODE={self.rag_mode!r}: use um de {list(RAG_MODES)}")
        if not isinstance(self.retrieval_k, int) or isinstance(self.retrieval_k, bool):
            errors.append(f"RAG_TOP_K={self.retrieval_k!r}: deve ser inteiro")
        elif not TOP_K_RANGE[0] <= self.retrieval_k <= TOP_K_RANGE[1]:
            errors.append(f"RAG_TOP_K={self.retrieval_k}: faixa aceita {TOP_K_RANGE[0]}..{TOP_K_RANGE[1]}")
        if not _is_number(self.min_score) or not MIN_SCORE_RANGE[0] <= self.min_score <= MIN_SCORE_RANGE[1]:
            errors.append(f"RAG_MIN_SCORE={self.min_score!r}: faixa aceita {MIN_SCORE_RANGE[0]}..{MIN_SCORE_RANGE[1]}")
        if (
            not _is_number(self.vector_only_min_score)
            or not MIN_SCORE_RANGE[0] <= self.vector_only_min_score <= MIN_SCORE_RANGE[1]
        ):
            errors.append(
                f"RAG_VECTOR_ONLY_MIN_SCORE={self.vector_only_min_score!r}: faixa aceita "
                f"{MIN_SCORE_RANGE[0]}..{MIN_SCORE_RANGE[1]}"
            )
        for env_name, value in (
            ("OLLAMA_EMBED_TIMEOUT_S", self.embed_timeout_s),
            ("OLLAMA_GENERATE_TIMEOUT_S", self.generate_timeout_s),
        ):
            if not _is_number(value) or not TIMEOUT_RANGE[0] <= value <= TIMEOUT_RANGE[1]:
                errors.append(f"{env_name}={value!r}: faixa aceita {TIMEOUT_RANGE[0]:g}..{TIMEOUT_RANGE[1]:g} segundos")
        if (
            not isinstance(self.embed_batch_size, int)
            or isinstance(self.embed_batch_size, bool)
            or not BATCH_SIZE_RANGE[0] <= self.embed_batch_size <= BATCH_SIZE_RANGE[1]
        ):
            errors.append(
                f"OLLAMA_EMBED_BATCH_SIZE={self.embed_batch_size!r}: faixa aceita {BATCH_SIZE_RANGE[0]}..{BATCH_SIZE_RANGE[1]}"
            )
        if self.rag_mode == "ollama":
            if _split_host_port(self.ollama_base_url) is None:
                errors.append(f"OLLAMA_BASE_URL={self.ollama_base_url!r}: informe uma URL http(s) completa")
            if not self.embedding_model.strip():
                errors.append("OLLAMA_EMBED_MODEL: não pode ser vazio em RAG_MODE=ollama")
            if not self.generation_model.strip():
                errors.append("OLLAMA_CHAT_MODEL: não pode ser vazio em RAG_MODE=ollama")
        if errors:
            raise ConfigError("Configuração inválida: " + "; ".join(errors))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        source: dict[str, str] = {}

        def read(name: str, default: str) -> str:
            raw = env.get(name)
            if raw is None or not raw.strip():
                source[name] = "default"
                return default
            source[name] = "env"
            return raw.strip()

        mode = read("RAG_MODE", "local").lower()
        return cls(
            rag_mode=mode,
            ollama_base_url=read("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            embedding_model=read("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe"),
            generation_model=read("OLLAMA_CHAT_MODEL", "qwen3:0.6b"),
            retrieval_k=_parse_int("RAG_TOP_K", read("RAG_TOP_K", "5")),
            min_score=_parse_float("RAG_MIN_SCORE", read("RAG_MIN_SCORE", "0.12")),
            vector_only_min_score=_parse_float("RAG_VECTOR_ONLY_MIN_SCORE", read("RAG_VECTOR_ONLY_MIN_SCORE", "0.5")),
            embed_timeout_s=_parse_float("OLLAMA_EMBED_TIMEOUT_S", read("OLLAMA_EMBED_TIMEOUT_S", "30")),
            generate_timeout_s=_parse_float("OLLAMA_GENERATE_TIMEOUT_S", read("OLLAMA_GENERATE_TIMEOUT_S", "60")),
            embed_batch_size=_parse_int("OLLAMA_EMBED_BATCH_SIZE", read("OLLAMA_EMBED_BATCH_SIZE", "32")),
            source=source,
        )

    def public_dict(self) -> dict[str, object]:
        """Valores seguros para log/diagnóstico (sem a URL do Ollama, que pode conter credenciais)."""
        host, port = _split_host_port(self.ollama_base_url) or (None, None)
        return {
            "rag_mode": self.rag_mode,
            "ollama_host": host,
            "ollama_port": port,
            "embedding_model": self.embedding_model if self.rag_mode == "ollama" else "hash-local",
            "generation_model": self.generation_model if self.rag_mode == "ollama" else "extractive-local",
            "retrieval_k": self.retrieval_k,
            "min_score": self.min_score,
            "vector_only_min_score": self.vector_only_min_score,
            "embed_timeout_s": self.embed_timeout_s,
            "generate_timeout_s": self.generate_timeout_s,
            "embed_batch_size": self.embed_batch_size,
        }


def _split_host_port(base_url: str) -> tuple[str, int | None] | None:
    """``(host, porta)`` de uma URL http(s) válida; ``None`` se a URL não servir para o cliente HTTP."""
    try:
        url = urlsplit(base_url)
        host, port = url.hostname, url.port  # ``port`` levanta ValueError se não for numérica
    except ValueError:
        return None
    if url.scheme not in {"http", "https"} or not host:
        return None
    return host, port


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value == value  # exclui NaN


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Configuração inválida: {name}={raw!r}: deve ser um inteiro") from exc


def _parse_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"Configuração inválida: {name}={raw!r}: deve ser um número") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ConfigError(f"Configuração inválida: {name}={raw!r}: deve ser um número finito")
    return value
