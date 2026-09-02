"""Cliente HTTP compartilhado e limite de concorrência para o Ollama (Fase 2, R-19/G-13 parte).

- ``SharedClient``: um ``httpx.Client`` por provider, criado na primeira chamada (lazy) e reutilizado com
  keep-alive; ``close()`` no shutdown. A criação lazy mantém compatível o padrão dos testes que
  substituem ``httpx.Client`` por um ``MockTransport``.
- ``OllamaGate``: semáforo de processo que limita chamadas simultâneas ao Ollama
  (``OLLAMA_MAX_CONCURRENCY``). Em CPU, um único modelo atende as requisições em série; sem o limite,
  N requisições simultâneas viram N timeouts em vez de N respostas enfileiradas. O tempo de espera na
  fila é medido e vai para o log (``queue_wait_ms``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

DEFAULT_MAX_CONCURRENCY = 2


class SharedClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    def get(self) -> httpx.Client:
        with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None and not self._client.is_closed:
                self._client.close()
            self._client = None


class OllamaGate:
    """Semáforo compartilhado por todos os providers de um mesmo ``RAGService``."""

    def __init__(self, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._waiting = 0
        self._lock = threading.Lock()

    @property
    def waiting(self) -> int:
        return self._waiting

    @contextmanager
    def acquire(self) -> Iterator[float]:
        """Bloqueia até haver vaga; devolve o tempo de espera em ms."""
        started = time.perf_counter()
        with self._lock:
            self._waiting += 1
        try:
            self._semaphore.acquire()
        finally:
            with self._lock:
                self._waiting -= 1
        try:
            yield round((time.perf_counter() - started) * 1000.0, 2)
        finally:
            self._semaphore.release()


_NO_GATE = OllamaGate(max_concurrency=1 << 16)  # efetivamente ilimitado (providers usados fora do serviço)


def no_gate() -> OllamaGate:
    return _NO_GATE
