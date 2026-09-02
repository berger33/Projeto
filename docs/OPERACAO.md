# Guia de operação

Referência rápida para quem roda o Aurora Document RAG. Detalhes de arquitetura em [`ARQUITETURA.md`](ARQUITETURA.md); decisões em [`DECISOES.md`](DECISOES.md).

## Variáveis de ambiente

Todas são validadas no boot; valor fora da faixa impede o processo de subir com a variável citada no log (`startup.failed`). Modelo em [`.env.example`](../.env.example).

| Variável | Padrão | Faixa / valores | Efeito |
|---|---|---|---|
| `RAG_MODE` | `local` | `local`, `ollama` | `local` = hash + extrativo (harness de testes); `ollama` = embeddings e LLM reais |
| `CORPUS_DIR` | `corpus` | caminho | diretório com `.pdf`, `.csv`, `.md`, `.txt` (relativo à raiz do projeto quando não absoluto) |
| `RAG_INDEX_DIR` | `.rag_index` | caminho ou vazio | índice persistido (`vectors.npy`, `chunks.json`, `manifest.json`); vazio desliga a persistência |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | URL http(s) | servidor Ollama (validada só em `RAG_MODE=ollama`) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | nome do modelo | prefixos de tarefa aplicados automaticamente por família |
| `OLLAMA_CHAT_MODEL` | `qwen3:1.7b` | nome do modelo | geração via `/api/chat`, `think: false`, saída JSON |
| `OLLAMA_EMBED_TIMEOUT_S` / `OLLAMA_GENERATE_TIMEOUT_S` | `30` / `60` | 1..600 | timeouts das chamadas |
| `OLLAMA_EMBED_BATCH_SIZE` | `32` | 1..256 | textos por requisição na indexação |
| `OLLAMA_NUM_CTX` / `OLLAMA_NUM_PREDICT` | `4096` / `300` | 1024..131072 / 32..4096 (`ctx − predict ≥ 512`) | janela e limite de geração; o prompt usa `ctx − predict − 64` |
| `OLLAMA_MAX_CONCURRENCY` | `2` | 1..64 | chamadas simultâneas ao Ollama (as demais esperam na fila) |
| `RAG_TOP_K` | `5` | 1..50 | trechos entregues ao gerador |
| `RAG_MIN_SCORE`, `RAG_VECTOR_ONLY_MIN_SCORE`, `RAG_VECTOR_WITH_OVERLAP_MIN_SCORE`, `RAG_MIN_LEXICAL_COVERAGE`, `RAG_HIGH_CONFIDENCE_SCORE`, `RAG_RELATIVE_GAP`, `RAG_MMR_LAMBDA` | perfil do modo | 0..1 | sobrescrevem o perfil de limiares (`THRESHOLD_PROFILES`) |
| `RAG_RERANKER` | `noop` | `noop` | reranker opcional (interface pronta; nenhum real incluído) |
| `RAG_CACHE_MAX_ENTRIES` / `RAG_CACHE_TTL_S` | `256` / `600` | 0..100000 / 0..86400 | cache de respostas (0 desliga) |
| `API_TOKEN` | vazio | ≥ 16 caracteres | exige `Authorization: Bearer` em `/api/*` |
| `RAG_RATE_LIMIT_PER_MINUTE` / `RAG_RATE_LIMIT_BURST` | `0` / = limite | 0..100000 / 1..100000 | token bucket por IP em `POST /api/ask` |
| `RAG_TRUST_PROXY` | `false` | `true`/`false` | honra `X-Forwarded-For` |
| `RAG_DOCS_ENABLED` | `true` | `true`/`false` | expõe `/docs`, `/redoc`, `/openapi.json` |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | níveis do `logging` / `json`, `text` | `DEBUG` inclui pergunta e resposta em texto |

## Modelos (Ollama, CPU)

Hardware de referência: i5-1135G7 (4 núcleos / 8 threads), 16 GB, sem GPU dedicada.

```bash
ollama pull nomic-embed-text-v2-moe   # embeddings multilíngues (~1 GB)
ollama pull qwen3:1.7b                # geração com recusa fundamentada em PT-BR
```

Alternativas: `embeddinggemma` (mais leve, prefixos próprios aplicados automaticamente), `bge-m3` (melhor com GPU); `qwen3:4b` melhora a qualidade a ~2× o tempo de prefill. Ao trocar de modelo de embedding, o manifesto detecta a mudança e o índice é reconstruído no próximo boot.

`GET /ready` confirma servidor e modelos antes de rotear tráfego.

## Índice e reindexação

```bash
python -m app.ingest             # constrói/atualiza o índice persistido (idempotente)
python -m app.ingest --check     # exit 1 se corpus/modelo/chunking mudaram desde a última indexação
python -m app.ingest --force     # reconstrói sempre
```

O boot compara o manifesto com o estado atual; se nada mudou, carrega do disco em milissegundos (sem chamar o Ollama). Qualquer alteração de arquivo, modelo, prefixo ou chunking reconstrói e registra o motivo em `index.rebuilt`. `RAGService.reload()` faz o mesmo em processo.

## Avaliação e calibração

```bash
python -m evals.run --mode local                 # 53 casos sobre corpus/ (roda também no pytest, com pisos)
python -m evals.run --mode ollama --save         # modo principal; grava evals/results/<ts>-ollama.json
python -m evals.calibrate --mode ollama          # varre limiares e imprime recusa correta × indevida
RAG_EVAL_OLLAMA=1 pytest -m ollama               # gate com os critérios de aceite do plano
```

Pisos do modo local e perfis de limiares ficam em [`evals/thresholds.json`](../evals/thresholds.json). **Pendente:** o perfil `ollama` é provisório até ser calibrado com o modelo real.

## Diagnóstico

Eventos JSON em stdout, sempre com `request_id` dentro de uma requisição. Os mais úteis:

- `index.built` / `index.loaded` / `index.rebuilt` (`reason`) — boot e persistência;
- `ingest.file` / `ingest.skipped` / `ingest.duplicate` — por arquivo do corpus;
- `query.retrieved` (candidatos com `vector_rank`, `lexical_rank`, `coverage`, `rrf`) → `query.answered` (`status`, `confidence`, `refusal_reason`, `support`, `sources_reason`, `timings_ms`);
- `answer.refused` — por que uma resposta do modelo foi rebaixada a recusa;
- `provider.embed` / `provider.generate` (`queue_wait_ms`, tokens, `done_reason`) / `provider.retry` / `prompt.truncated`;
- `http.error` — detalhe interno de qualquer 4xx/5xx (o cliente só recebe `error_code`).

Correlacione um problema reportado pelo usuário pelo `request_id` devolvido no corpo/cabeçalho da resposta.

## Códigos de erro da API

| HTTP | `error_code` |
|---|---|
| 422 | `invalid_question` / validação do corpo |
| 401 | `unauthorized` |
| 429 | `rate_limited` (`Retry-After`) |
| 503 | `index_not_ready`, `provider_unavailable`, `provider_timeout`, `provider_invalid_response` |
| 500 | `internal_error` |
