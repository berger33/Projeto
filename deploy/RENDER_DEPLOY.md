# Alternativa de nuvem: Render

O repositório inclui `render.yaml` para uma publicação alternativa quando não houver VM disponível.

**Limitação importante:** o plano `free` do Render não executa Ollama (sem GPU, 512 MB de RAM, sem
serviços auxiliares). O blueprint publica a API em **modo `local`** (hash embedding + gerador
extrativo): serve para demonstrar contrato HTTP, retrieval híbrido, recusa e rastreabilidade das
fontes — **não** a geração com LLM. Para o modo `ollama`, use `docker-compose.yml` numa VM
(`OCI_DEPLOY.md`) ou defina `RAG_MODE=ollama` + `OLLAMA_BASE_URL` apontando para um servidor
Ollama acessível pela rede do Render.

## Passos

1. Conecte sua conta GitHub ao Render.
2. Crie um novo serviço usando o repositório `berger33/aurora-document-rag`.
3. Utilize o Blueprint definido em `render.yaml` (Docker, `healthCheckPath: /ready`, `RAG_MODE=local`).
4. Aguarde o build (a imagem pré-constrói o índice; o boot leva milissegundos).
5. Valide a URL pública em `/ready` (deve devolver `{"ok": true, ...}`) e depois `/health`.
6. Faça uma pergunta em `/` (interface servida de `app/static/`) ou via `POST /api/ask`.

## Endurecimento (opcional)

Como a URL fica pública, considere ligar no painel do Render (ou descomentando em `render.yaml`):

| Variável | Efeito |
|---|---|
| `API_TOKEN` (≥ 16 caracteres, como *secret*) | exige `Authorization: Bearer` em `/api/*` |
| `RAG_RATE_LIMIT_PER_MINUTE=30` | limite por IP em `POST /api/ask` (429 + `Retry-After`) |
| `RAG_TRUST_PROXY=true` | o Render está atrás de proxy; sem isso o IP visto é o do balanceador |
| `RAG_DOCS_ENABLED=false` | oculta `/docs`, `/redoc` e `/openapi.json` |
