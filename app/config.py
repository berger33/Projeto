"""Configuração da aplicação a partir de variáveis de ambiente, validada no boot.

Toda variável malformada ou fora de faixa gera ``ConfigError`` com o nome da variável e a faixa
aceita, de modo que a aplicação falhe ao subir (``lifespan``) em vez de na primeira requisição
(Fase 2, G-04). ``Settings`` continua um ``dataclass`` congelado: a validação também roda quando o
objeto é construído diretamente em código/testes.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

RAG_MODES = ("local", "ollama")

# (mínimo, máximo) inclusivos
TOP_K_RANGE = (1, 50)
MIN_SCORE_RANGE = (0.0, 1.0)
TIMEOUT_RANGE = (1.0, 600.0)
BATCH_SIZE_RANGE = (1, 256)
NUM_CTX_RANGE = (1024, 131072)
NUM_PREDICT_RANGE = (32, 4096)


class ConfigError(ValueError):
    """Configuração inválida. A mensagem é escrita para o operador e cita a variável de ambiente."""


@dataclass(frozen=True)
class RetrievalThresholds:
    """Limiares de retrieval/confiança de um provider de embeddings (Fase 2, R-11/R-25).

    A escala do cosseno depende do modelo: no hash local, legítimos ficam em 0,26-0,63 e irrelevantes
    até ~0,35; em modelos densos (nomic, bge) irrelevantes costumam ficar >= 0,3 e legítimos >= 0,55.
    Os valores default de cada perfil vêm de ``evals/thresholds.json`` (calibrados por
    ``python -m evals.calibrate``); variáveis de ambiente sobrescrevem individualmente.
    """

    min_score: float
    vector_only_min_score: float
    vector_with_overlap_min_score: float
    min_lexical_coverage: float
    high_confidence_score: float  # top-1 acima disto (com gap/concordância) => "alta"
    relative_gap: float  # (top1 - top2) / top1 mínimo para considerar o top-1 destacado
    mmr_lambda: float = 1.0  # diversificação MMR dos aprovados (1.0 = desligada)


# Perfis calibrados (ver evals/thresholds.json → "profiles"; mantidos aqui como fonte da verdade em código).
THRESHOLD_PROFILES: dict[str, RetrievalThresholds] = {
    "local": RetrievalThresholds(
        min_score=0.12,
        vector_only_min_score=0.5,
        vector_with_overlap_min_score=0.35,
        min_lexical_coverage=0.2,
        high_confidence_score=0.45,
        relative_gap=0.15,
        # Eval local (P2-05): MMR 0,7 dá +2,2 p.p. de selected recall mas -6,7 p.p. de precisão de fontes
        # com o hash (cosseno entre chunks é ruidoso) → desligado neste perfil.
        mmr_lambda=1.0,
    ),
    # Provisório até haver medição com o modelo real (P1-06 registra a dúvida; calibrar com
    # `python -m evals.calibrate --mode ollama` assim que houver Ollama disponível).
    "ollama": RetrievalThresholds(
        min_score=0.35,
        vector_only_min_score=0.65,
        vector_with_overlap_min_score=0.5,
        min_lexical_coverage=0.2,
        high_confidence_score=0.7,
        relative_gap=0.1,
        mmr_lambda=0.7,  # embeddings densos: quase-duplicatas reais (faq.csv x faq.pdf) têm cosseno alto
    ),
}


@dataclass(frozen=True)
class Settings:
    rag_mode: str = "local"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text-v2-moe"
    generation_model: str = "qwen3:1.7b"
    retrieval_k: int = 5
    # Limiares: ``None`` = usar o perfil do modo (THRESHOLD_PROFILES[rag_mode]); env sobrescreve.
    min_score: float | None = None
    vector_only_min_score: float | None = None
    vector_with_overlap_min_score: float | None = None
    min_lexical_coverage: float | None = None
    high_confidence_score: float | None = None
    relative_gap: float | None = None
    embed_timeout_s: float = 30.0
    generate_timeout_s: float = 60.0
    embed_batch_size: int = 32
    # Janela de contexto e limite de geração enviados ao Ollama (options.num_ctx / num_predict).
    num_ctx: int = 4096
    num_predict: int = 300
    # Diversificação MMR dos trechos aprovados (0..1; 1 desliga): None = perfil do modo. Reranker opcional.
    mmr_lambda: float | None = None
    reranker: str = "noop"
    # Diretório do índice persistido (.npy + manifest). "" desabilita a persistência (reembeda a cada boot).
    # O default lê RAG_INDEX_DIR mesmo em construção direta, para que testes/CLIs isolem o índice.
    index_dir: str = field(default_factory=lambda: _read_index_dir(os.environ))
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
        for env_name, score in (
            ("RAG_MIN_SCORE", self.min_score),
            ("RAG_VECTOR_ONLY_MIN_SCORE", self.vector_only_min_score),
            ("RAG_VECTOR_WITH_OVERLAP_MIN_SCORE", self.vector_with_overlap_min_score),
            ("RAG_MIN_LEXICAL_COVERAGE", self.min_lexical_coverage),
            ("RAG_HIGH_CONFIDENCE_SCORE", self.high_confidence_score),
            ("RAG_RELATIVE_GAP", self.relative_gap),
        ):
            if score is None:
                continue
            if not _is_number(score) or not MIN_SCORE_RANGE[0] <= score <= MIN_SCORE_RANGE[1]:
                errors.append(f"{env_name}={score!r}: faixa aceita {MIN_SCORE_RANGE[0]}..{MIN_SCORE_RANGE[1]}")
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
        if self.mmr_lambda is not None and (not _is_number(self.mmr_lambda) or not 0.0 <= self.mmr_lambda <= 1.0):
            errors.append(f"RAG_MMR_LAMBDA={self.mmr_lambda!r}: faixa aceita 0.0..1.0")
        if not self.reranker.strip():
            errors.append("RAG_RERANKER: não pode ser vazio (use 'noop')")
        for env_name, value, bounds in (
            ("OLLAMA_NUM_CTX", self.num_ctx, NUM_CTX_RANGE),
            ("OLLAMA_NUM_PREDICT", self.num_predict, NUM_PREDICT_RANGE),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not bounds[0] <= value <= bounds[1]:
                errors.append(f"{env_name}={value!r}: faixa aceita {bounds[0]}..{bounds[1]}")
        if (
            isinstance(self.num_ctx, int)
            and isinstance(self.num_predict, int)
            and self.num_ctx - self.num_predict < 512
        ):
            errors.append(
                "OLLAMA_NUM_CTX deve exceder OLLAMA_NUM_PREDICT em pelo menos 512 tokens (espaço para o prompt)"
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
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env: Mapping[str, str] = os.environ if environ is None else environ
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
            generation_model=read("OLLAMA_CHAT_MODEL", "qwen3:1.7b"),
            retrieval_k=_parse_int("RAG_TOP_K", read("RAG_TOP_K", "5")),
            min_score=_parse_optional_float("RAG_MIN_SCORE", env),
            vector_only_min_score=_parse_optional_float("RAG_VECTOR_ONLY_MIN_SCORE", env),
            vector_with_overlap_min_score=_parse_optional_float("RAG_VECTOR_WITH_OVERLAP_MIN_SCORE", env),
            min_lexical_coverage=_parse_optional_float("RAG_MIN_LEXICAL_COVERAGE", env),
            high_confidence_score=_parse_optional_float("RAG_HIGH_CONFIDENCE_SCORE", env),
            relative_gap=_parse_optional_float("RAG_RELATIVE_GAP", env),
            embed_timeout_s=_parse_float("OLLAMA_EMBED_TIMEOUT_S", read("OLLAMA_EMBED_TIMEOUT_S", "30")),
            generate_timeout_s=_parse_float("OLLAMA_GENERATE_TIMEOUT_S", read("OLLAMA_GENERATE_TIMEOUT_S", "60")),
            embed_batch_size=_parse_int("OLLAMA_EMBED_BATCH_SIZE", read("OLLAMA_EMBED_BATCH_SIZE", "32")),
            num_ctx=_parse_int("OLLAMA_NUM_CTX", read("OLLAMA_NUM_CTX", "4096")),
            num_predict=_parse_int("OLLAMA_NUM_PREDICT", read("OLLAMA_NUM_PREDICT", "300")),
            mmr_lambda=_parse_optional_float("RAG_MMR_LAMBDA", env),
            reranker=read("RAG_RERANKER", "noop").lower(),
            index_dir=_read_index_dir(env),
            source=source,
        )

    @property
    def thresholds(self) -> RetrievalThresholds:
        """Perfil do modo com as sobrescritas explícitas aplicadas."""
        profile = THRESHOLD_PROFILES[self.rag_mode]
        overrides = {
            name: value
            for name, value in (
                ("min_score", self.min_score),
                ("vector_only_min_score", self.vector_only_min_score),
                ("vector_with_overlap_min_score", self.vector_with_overlap_min_score),
                ("min_lexical_coverage", self.min_lexical_coverage),
                ("high_confidence_score", self.high_confidence_score),
                ("relative_gap", self.relative_gap),
                ("mmr_lambda", self.mmr_lambda),
            )
            if value is not None
        }
        return dataclasses.replace(profile, **overrides) if overrides else profile

    def public_dict(self) -> dict[str, object]:
        """Valores seguros para log/diagnóstico (sem a URL do Ollama, que pode conter credenciais)."""
        host, port = _split_host_port(self.ollama_base_url) or (None, None)
        thresholds = self.thresholds
        return {
            "rag_mode": self.rag_mode,
            "ollama_host": host,
            "ollama_port": port,
            "embedding_model": self.embedding_model if self.rag_mode == "ollama" else "hash-local",
            "generation_model": self.generation_model if self.rag_mode == "ollama" else "extractive-local",
            "retrieval_k": self.retrieval_k,
            "thresholds": dataclasses.asdict(thresholds),
            "embed_timeout_s": self.embed_timeout_s,
            "generate_timeout_s": self.generate_timeout_s,
            "embed_batch_size": self.embed_batch_size,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "reranker": self.reranker,
            "index_dir": self.index_dir or None,
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


def _read_index_dir(env: Mapping[str, str]) -> str:
    """``RAG_INDEX_DIR`` ausente → padrão; definido como vazio → persistência desligada."""
    raw = env.get("RAG_INDEX_DIR")
    return ".rag_index" if raw is None else raw.strip()


def _parse_optional_float(name: str, env: Mapping[str, str]) -> float | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return _parse_float(name, raw.strip())


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
