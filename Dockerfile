# syntax=docker/dockerfile:1.7
# Imagem de runtime do Aurora Document RAG (P3-02).
# - Apenas dependências de runtime travadas pelo lockfile (requirements.txt é exportado de uv.lock).
# - Copia só o necessário (app/ + docs/); .dockerignore exclui .git, .env, testes, auditoria etc.
# - Usuário não-root; índice persistido em volume próprio (RAG_INDEX_DIR=/data/index).
# - O índice é pré-construído no build para o modo local (boot em milissegundos); no modo ollama
#   ele é (re)construído no primeiro boot contra o servidor Ollama e persistido no volume.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_INDEX_DIR=/data/index \
    RAG_MODE=local

# curl só para o HEALTHCHECK (imagem slim não o traz).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 aurora \
    && useradd --system --uid 10001 --gid aurora --home-dir /app --shell /usr/sbin/nologin aurora \
    && mkdir -p /app /data/index \
    && chown -R aurora:aurora /app /data

WORKDIR /app

COPY --chown=aurora:aurora requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=aurora:aurora app/ ./app/
COPY --chown=aurora:aurora docs/ ./docs/

USER aurora

# Pré-indexa o corpus para o modo local (hash). Em modo ollama o manifesto não bate (modelo diferente)
# e o índice é reconstruído no boot — comportamento esperado e logado (index.rebuilt).
RUN python -m app.ingest --index-dir /data/index

VOLUME ["/data/index"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
