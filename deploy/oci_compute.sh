#!/usr/bin/env bash
# Sobe a API em modo local (sem Ollama) numa VM Ubuntu da OCI Compute com Docker instalado.
# Para o modo ollama (API + LLM) use `docker compose up -d --build` — ver deploy/OCI_DEPLOY.md.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/berger33/aurora-document-rag.git}"
APP_DIR="${APP_DIR:-aurora-document-rag}"
IMAGE="${IMAGE:-aurora-document-rag}"
CONTAINER="${CONTAINER:-aurora-rag}"
HOST_PORT="${HOST_PORT:-80}"

if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
git pull --ff-only

sudo docker build -t "$IMAGE" .
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# RAG_MODE=local é o padrão da imagem; o índice já vem pré-construído em /data/index.
# Variáveis opcionais (API_TOKEN, RAG_RATE_LIMIT_PER_MINUTE, ...) podem vir de um .env na raiz.
ENV_FILE_ARG=()
if [ -f .env ]; then
  ENV_FILE_ARG=(--env-file .env)
fi
sudo docker run -d --name "$CONTAINER" --restart unless-stopped \
  -p "${HOST_PORT}:8000" "${ENV_FILE_ARG[@]}" "$IMAGE"

# /ready só responde 200 quando o índice está carregado.
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:${HOST_PORT}/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://localhost:${HOST_PORT}/ready"
echo
