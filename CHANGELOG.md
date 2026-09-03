# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versionamento semântico.
Os identificadores `P0-xx`…`P3-xx` referem-se aos itens do plano de ação da auditoria
(`auditoria/fase-3-plano-de-acao.md`); `G-xx`/`R-xx` são os achados de `auditoria/fase-2-relatorio.md`.

## [2.1.0] — 2026-09-02

Resultado da auditoria técnica em quatro fases (mapeamento, achados, plano, implementação). O contrato
de `POST /api/ask` foi estendido **apenas de forma aditiva** (D10): clientes da 2.0.0 continuam funcionando.

### Adicionado

- **Observabilidade (P0-02):** logs JSON estruturados, `X-Request-ID` gerado/propagado, `timings_ms` por
  etapa (`retrieve`, `filter`, `generate`, `verify`) na resposta e nos eventos.
- **Avaliação (P0-03, P1-06):** harness `python -m evals.run` sobre o corpus real (53 casos em 7
  categorias; Recall@k, MRR, selected recall, precisão de fontes, recusa correta/indevida, content pass,
  latência), gate de regressão em `tests/test_evals.py` com pisos em `evals/thresholds.json`,
  `python -m evals.calibrate` para varrer limiares por provider.
- **Ciclo de vida (P0-04):** índice construído no `lifespan`; `GET /ready` (índice + Ollama + modelos);
  `Settings` validado no boot (`ConfigError` derruba o processo com `startup.failed`); contrato de erro
  uniforme `{detail, error_code, request_id}` (422/401/429/503/500) sem vazar detalhes internos.
- **Recusa estruturada (P1-01):** `status` (`answered` / `refused_no_context` / `refused_by_model`),
  `refusal_reason`, `support`; gerador Ollama responde JSON (`answer`, `grounded`, `used_sources`);
  juiz de recusa em PT-BR (declaração, padrão, sustentação lexical, números não sustentados).
- **Embeddings (P1-02):** prefixos de tarefa por família de modelo (`search_query:`/`search_document:`
  para nomic, etc.), lotes com retry/backoff, validação de dimensão, padrão multilíngue
  `nomic-embed-text-v2-moe`.
- **Chunking por seção (P1-03):** `app/chunking.py` com títulos numerados, remoção de cabeçalho/rodapé
  repetidos, orçamento de ~300 tokens, `section`/`char_start`/`char_end`/`token_estimate` por chunk.
- **Retrieval híbrido (P1-04):** BM25 sobre texto normalizado (acentos, stopwords PT-BR, radicais) +
  cosseno, fusão RRF, três níveis de evidência (cosseno alto; cosseno médio + termo em comum; cobertura
  lexical) antes do corte em k.
- **Prompt com orçamento (P1-05):** `/api/chat` com `think:false`, `format` JSON, `num_ctx`/`num_predict`
  explícitos, trechos e pergunta em blocos delimitados com `<` escapado, evento `prompt.truncated`.
- **Fontes rastreáveis (P2-01):** `chunk_id`, `score`, `section`, `excerpt` e `inferred` em cada fonte;
  fontes derivadas do que o gerador declarou usar (`used_sources` ou marcadores `[n]`).
- **Ingestão (P2-02):** Markdown e texto além de PDF/CSV; CSV via biblioteca padrão (delimitador
  detectado, BOM, tudo como texto); `IngestReport` por arquivo (`ingest.file`/`skipped`/`duplicate`/
  `empty`); dedup de quase-duplicatas (Jaccard de radicais ≥ 0,9); `IngestError` no boot para arquivo
  ilegível.
- **Busca vetorial numpy (P2-04):** `VectorStore` plugável e `NumpyVectorStore` (matriz normalizada,
  top-k por `argpartition`, filtro por metadado): < 1 ms para 10 mil chunks × 768 dims.
- **Índice persistido (P2-03):** `.npy` + `chunks.json` + `manifest.json` em `RAG_INDEX_DIR`
  (padrão `.rag_index/`), invalidação por hash dos arquivos/modelo/dimensão/versão de chunking,
  `python -m app.ingest [--check|--force]`, `RAGService.reload()`; boot com índice pronto em ~20 ms.
- **MMR e Reranker (P2-05):** diversificação dos aprovados (`mmr_lambda` por perfil: 1.0 local,
  0.7 ollama) e interface `Reranker` (`RAG_RERANKER`, hoje só `noop`).
- **Cache e concorrência (P2-06):** cache LRU+TTL de respostas com chave normalizada + versão do índice
  e do prompt (`RAG_CACHE_MAX_ENTRIES`, `RAG_CACHE_TTL_S`), semáforo para o Ollama
  (`OLLAMA_MAX_CONCURRENCY`, `queue_wait_ms` nos eventos), cliente HTTP reutilizado e fechado no shutdown.
- **Segurança opcional (P3-03):** `API_TOKEN` (Bearer em `/api/*`), `RAG_RATE_LIMIT_PER_MINUTE`/`_BURST`
  (token bucket por IP, 429 + `Retry-After`), `RAG_TRUST_PROXY`, `RAG_DOCS_ENABLED`; tudo desligado
  por padrão (D5).
- **Interface (P3-04):** página em `/` servida de `app/static/` como cliente fino da API (status,
  confiança, fontes com trecho, `request_id`).
- **Documentação (P3-05, P3-06):** `docs/ARQUITETURA.md`, `docs/OPERACAO.md`, `docs/DECISOES.md`,
  seção "Limitações" no README, `SECURITY.md`, este changelog; guias de deploy atualizados.
- **CI (P3-01):** matriz Python 3.11/3.12/3.13, `mypy --strict` em `app/` e `evals/`, cobertura ≥ 85 %,
  eval local como gate, `app.ingest --check`, `pip-audit`, `uv lock --check`, verificação real de
  whitespace no diff do PR, build da imagem + boot + `/ready` + pergunta real.

### Alterado

- **Empacotamento (P0-01):** `pyproject.toml` + `uv.lock`; `requirements.txt` (runtime) e
  `requirements-dev.txt` (ferramentas) exportados do lock; `pytest` roda da raiz sem `PYTHONPATH`.
- **Modelos padrão:** embedding `nomic-embed-text` → `nomic-embed-text-v2-moe` (o original é só inglês);
  geração → `qwen3:1.7b` com `think:false` (D1/D2). Reindexação automática na troca.
- **Ordem do pipeline:** filtros de evidência e limiares por perfil (`THRESHOLD_PROFILES`) rodam sobre
  o pool fundido antes do corte em k; `confidence` combina score do top-1, gap relativo ou concordância
  de documentos e sustentação medida.
- **Corpus:** `docs/` → `corpus/` (`CORPUS_DIR`); `docs/` passa a conter documentação (D9).
- **Docker (P3-02):** imagem não-root, apenas `app/` + `corpus/` + `requirements.txt` copiados
  (`.dockerignore` em lista branca), índice pré-construído, `HEALTHCHECK`; `docker-compose.yml` com
  `ollama` + `ollama-pull` + volumes.
- **Deploy (P3-06):** `render.yaml` com `healthCheckPath: /ready` e `envVars` explícitos (modo `local`;
  o plano free não roda Ollama); `deploy/*` aponta para `berger33/aurora-document-rag` e documenta os
  dois caminhos (compose com Ollama ou só a API).
- **Scripts Windows (P3-06):** `DIAGNOSTICO_WINDOWS.bat` importa só dependências reais
  (`fastapi`, `pypdf`, `httpx`, `numpy`) e verifica o índice persistido, em vez de falhar sempre por
  importar `langchain`/`pandas` (G-24).
- README reescrito para descrever o que existe (sem overclaims — G-30), com tabela de evidências,
  diagrama atual e limitações.

### Removido

- `pandas`/`pandas-stubs` das dependências (CSV via `csv` da biblioteca padrão).
- `demo/index.html` (D7): a demo estática divergia do backend; a interface em `/` consome a API real.
- HTML embutido em `app/main.py`.

### Corrigido

- Recusas do modelo recebiam fontes (10/12 formulações antes; 0/20 depois) — R-13.
- `nomic-embed-text` usado sem os prefixos obrigatórios e com corpus em português — R-08/R-09.
- Chunks vazios/duplicados e cabeçalhos repetidos no índice — R-03.
- Erros do Ollama retornavam 500 com a mensagem interna; agora 503 + `error_code` estável — G-01/G-03.
- Perguntas sem acento ou com flexão não recuperavam nada (canal lexical inexistente) — R-10.
- `X-Forwarded-For` nunca deve ser confiado sem proxy declarado; `Retry-After` correto no 429.
- Passo de whitespace da CI era um no-op (comparava `HEAD` com `HEAD`) — G-22.

### Métricas (modo local, 53 casos)

| Métrica | Baseline (P0-03) | 2.1.0 |
|---|---|---|
| Recall@5 | 0,93 | 1,00 |
| MRR | 0,73 | 0,82 |
| Selected recall | — | 0,82 |
| Precisão de fontes | 0,71 | 0,90 |
| Recusa correta (fora de escopo) | 0,50 | 1,00 |
| Recusa indevida | — | 0,05 |
| Content pass | — | 0,81 |
| Cobertura de testes (branch) | 75 % | 96 % (piso 85 %) |
| Testes | 8 | 400+ |

Números do modo `ollama` ainda não foram medidos (ver "Limitações" no README e `docs/DECISOES.md`).

## [2.0.0] — 2026-08

Reestruturação inicial: FastAPI, ingestão PDF/CSV, hash embedding local, provider Ollama, modo
extrativo, evals iniciais, CI básica. Estado avaliado pela auditoria (`auditoria/fase-2-relatorio.md`).

[2.1.0]: https://github.com/berger33/aurora-document-rag/compare/1183cf4...main
[2.0.0]: https://github.com/berger33/aurora-document-rag/commit/1183cf4
