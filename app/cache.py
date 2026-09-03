"""Cache LRU/TTL de respostas em memória (Fase 2, R-19).

Chave: ``(pergunta normalizada, versão do índice, versão do prompt, modo)``. A versão do índice vem do
manifesto (hash do conjunto de arquivos + modelo + chunking), então um ``reload()`` ou uma
reindexação invalidam automaticamente todas as entradas; a versão do prompt protege contra respostas
geradas por um template antigo. Entradas expiram por TTL e o tamanho é limitado por LRU.

Só respostas ``answered`` e recusas por falta de contexto são cacheadas — recusas do modelo
(``refused_by_model``) e erros não, porque podem depender de variação do LLM.

Thread-safe (lock simples): o servidor executa endpoints síncronos num threadpool.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .domain import RAGRun
from .text import tokenize


def cache_key(question: str, *, index_version: str, prompt_version: str, mode: str) -> str:
    """Chave insensível a caixa, acentos, pontuação e espaçamento (mesma tokenização do pipeline)."""
    raw = f"{mode}\x1f{index_version}\x1f{prompt_version}\x1f{' '.join(tokenize(question))}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "hit_rate": self.hit_rate,
        }


@dataclass
class _Entry:
    run: RAGRun
    expires_at: float


@dataclass
class AnswerCache:
    max_entries: int = 256
    ttl_s: float = 600.0
    _entries: OrderedDict[str, _Entry] = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    stats: CacheStats = field(default_factory=CacheStats)

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_s > 0

    def get(self, key: str) -> RAGRun | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if entry.expires_at <= time.monotonic():
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                self.stats.size = len(self._entries)
                return None
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return entry.run

    def put(self, key: str, run: RAGRun) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = _Entry(run=run, expires_at=time.monotonic() + self.ttl_s)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1
            self.stats.size = len(self._entries)

    def clear(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self.stats.size = 0
            return removed
