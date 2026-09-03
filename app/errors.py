"""Exceções de domínio com mapeamento estável para HTTP.

Cada classe define ``status_code``, ``error_code`` (contrato público, estável) e ``public_detail``
(mensagem genérica devolvida ao cliente). A mensagem passada ao construtor é **interna**: vai para os
logs junto com o ``request_id`` e nunca para a resposta HTTP (Fase 2, G-02).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx


class AuroraError(Exception):
    status_code = 500
    error_code = "internal_error"
    public_detail = "Erro interno ao processar a requisição."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_detail)


class InvalidQuestionError(AuroraError, ValueError):
    """Pergunta rejeitada pelo domínio. A mensagem é escrita para o usuário e pode ser exposta."""

    status_code = 422
    error_code = "invalid_question"
    public_detail = "A pergunta é inválida."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)
        if message:
            self.public_detail = message


class IndexNotReadyError(AuroraError):
    status_code = 503
    error_code = "index_not_ready"
    public_detail = "O índice de documentos ainda não está pronto."


class ProviderError(AuroraError):
    """Falha ao usar um provider externo (Ollama). Detalhes (URL, erro de rede) só no log."""

    status_code = 503
    error_code = "provider_error"
    public_detail = "Serviço de resposta temporariamente indisponível. Tente novamente em instantes."


class ProviderUnavailableError(ProviderError):
    error_code = "provider_unavailable"


class ProviderTimeoutError(ProviderError):
    error_code = "provider_timeout"


class ProviderResponseError(ProviderError):
    error_code = "provider_invalid_response"


@contextmanager
def provider_call(operation: str, url: str) -> Iterator[None]:
    """Traduz falhas de transporte/HTTP/parse do ``httpx`` em ``ProviderError`` com contexto interno.

    Exceções já tipadas (``ProviderError``) passam intactas.
    """
    try:
        yield
    except ProviderError:
        raise
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(f"timeout em {operation} ({url}): {type(exc).__name__}: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300]
        raise ProviderUnavailableError(
            f"HTTP {exc.response.status_code} em {operation} ({url}): {body or exc.response.reason_phrase}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderUnavailableError(f"falha de conexão em {operation} ({url}): {type(exc).__name__}: {exc}") from exc
    except ValueError as exc:  # JSON inválido
        raise ProviderResponseError(f"resposta não decodificável em {operation} ({url}): {exc}") from exc


def ping_ollama(base_url: str, *, timeout: float = 2.0) -> list[str]:
    """Consulta ``GET /api/tags`` e devolve os nomes de modelos disponíveis. Levanta ``ProviderError``."""
    url = f"{base_url.rstrip('/')}/api/tags"
    with provider_call("tags", url), httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ProviderResponseError(f"/api/tags devolveu payload inesperado ({type(payload).__name__})")
    return [str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)]
