# Aurora Document RAG

**Python backend para atendimento documental com FastAPI, retrieval vetorial, geração local opcional e citações rastreáveis.**

[![CI](https://github.com/berger33/aurora-document-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/berger33/aurora-document-rag/actions/workflows/ci.yml)

> Projeto acadêmico do Challenge Alura, reestruturado como case de backend e IA aplicada. O objetivo é responder perguntas sobre políticas da Aurora Moda Online usando somente os documentos do repositório — e recusar quando não houver evidência suficiente.

## Evidência para avaliação técnica

| Evidência | Onde verificar |
|---|---|
| API tipada e validação | [`app/main.py`](app/main.py) |
| ingestão de PDF/CSV | [`app/documents.py`](app/documents.py) |
| providers de embeddings | [`app/embeddings.py`](app/embeddings.py) |
| índice vetorial + cosine similarity | [`app/retrieval.py`](app/retrieval.py) |
| prompt e geração | [`app/generation.py`](app/generation.py) |
| orquestração RAG | [`app/rag.py`](app/rag.py) |
| testes de domínio/API | [`tests/test_rag.py`](tests/test_rag.py) |
| evals versionados | [`evals/cases.json`](evals/cases.json) |
| arquitetura | [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) · operação: [`docs/OPERACAO.md`](docs/OPERACAO.md) · decisões: [`docs/DECISOES.md`](docs/DECISOES.md) |

## Arquitetura

```text
PDF / CSV
   ↓
Document loader + chunking
   ↓
Embedding provider
   ├─ local hash embedding (CI/offline)
   └─ Ollama / nomic-embed-text-v2-moe
   ↓
Vector index + cosine similarity
   ↓
Top-k + threshold + gate de relevância
   ↓
Answer generator
   ├─ fallback extrativo local
   └─ Ollama LLM / qwen3
   ↓
FastAPI → resposta + fontes + confiança + modo
```

O modo `ollama` é o caminho **RAG generativo**: documentos e pergunta viram embeddings, os trechos relevantes são recuperados e enviados ao LLM com instruções para responder apenas com o contexto. O modo `local` existe para CI e execução offline; ele é extrativo e não é apresentado como LLM.

## Executar

### Modo local reproduzível

Requer Python **3.11+**. As dependências são travadas em `uv.lock`; `requirements.txt` (runtime) e `requirements-dev.txt` (ferramentas) são exportados dele.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt            # runtime
pip install -r requirements-dev.txt        # opcional: pytest, ruff, mypy, coverage, pip-audit
uvicorn app.main:app --reload
```

Com [uv](https://docs.astral.sh/uv/) instalado, o equivalente é `uv sync` (cria `.venv` com runtime + dev) e `uv run uvicorn app.main:app --reload`.

Abra `http://127.0.0.1:8000` (interface mínima servida de `app/static/`, que consome a própria API) ou `/docs`.

### RAG generativo com Ollama

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull qwen3:1.7b
```

Configure:

```env
RAG_MODE=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=nomic-embed-text-v2-moe
OLLAMA_CHAT_MODEL=qwen3:1.7b
OLLAMA_EMBED_TIMEOUT_S=30        # opcional (1..600)
OLLAMA_GENERATE_TIMEOUT_S=60     # opcional (1..600); em CPU, modelos maiores podem precisar de mais
OLLAMA_EMBED_BATCH_SIZE=32       # opcional (1..256): textos por requisição na indexação
OLLAMA_NUM_CTX=4096              # opcional: janela de contexto enviada ao modelo
OLLAMA_NUM_PREDICT=300           # opcional: máximo de tokens gerados por resposta
```

#### Geração

A geração usa `/api/chat` com papel `system` separado, `think: false` (qwen3 não gasta tokens "pensando" e nada de `<think>` vaza para a resposta), saída JSON forçada por `format` e `options.num_ctx`/`num_predict` explícitos. O prompt tem **orçamento**: `num_ctx − num_predict − margem` tokens (estimados a 3,5 caracteres/token); os trechos entram inteiros, do mais relevante ao menos relevante, até o limite — o que não couber é registrado no evento `prompt.truncated` em vez de ser cortado silenciosamente pelo servidor. Trechos e pergunta são envolvidos em tags (`<contexto>`, `<fonte n="…">`, `<pergunta>`) cujo `<` é escapado dentro de conteúdo não confiável, para que um documento ou uma pergunta não consigam fechar ou abrir blocos do template. Resposta cortada por `num_predict` (`done_reason: length`) sem JSON completo é tratada como recusa. O padrão `qwen3:1.7b` é o menor modelo que responde com recusa fundamentada em PT-BR em CPU (i5 de 11ª geração, 16 GB); `qwen3:4b` melhora a qualidade a ~2× o tempo de prefill.

#### Modelo de embedding

O padrão é `nomic-embed-text-v2-moe` (multilíngue, ~1 GB, rápido em CPU). O `nomic-embed-text` original é treinado para inglês e não deve ser usado com corpus em português. Os **prefixos de tarefa** exigidos por cada família são aplicados automaticamente a partir do nome do modelo (`search_query:`/`search_document:` para nomic; `task: search result | query:`/`title: none | text:` para `embeddinggemma`; instrução de consulta para `qwen3-embedding` e `mxbai-embed-large`; nenhum para `bge-m3`). Para trocar de modelo basta alterar `OLLAMA_EMBED_MODEL` e reiniciar — o índice é reconstruído no boot e a dimensão dos vetores é validada; modelos fora da tabela funcionam sem prefixo.

A indexação envia os chunks em lotes (`OLLAMA_EMBED_BATCH_SIZE`) com até 2 novas tentativas e backoff em falhas transitórias (timeout, conexão, HTTP 5xx/429); erro definitivo (modelo não instalado, 404) falha imediatamente com mensagem clara.

`GET /ready` confirma se o servidor e os dois modelos estão disponíveis antes de enviar perguntas.

## API

`POST /api/ask`

```json
{"question":"Qual é o prazo para devolução?"}
```

A resposta contém `answer`, `sources`, `confidence`, `mode`, `request_id`, `timings_ms` (ms por etapa: `retrieve`, `filter`, `generate`, `verify`), `status` e `refusal_reason`.

Cada fonte traz `document`, `page`/`row`, e ainda `chunk_id` (trecho exato do índice), `score` (cosseno), `section` (título detectado no documento) e `excerpt` (a frase do trecho que mais sustenta a resposta, ≤ 200 caracteres). As fontes vêm do que o gerador **declarou usar** (`used_sources` da saída JSON ou marcadores `[n]` no texto, que são removidos da resposta); só quando não há declaração as fontes caem para os trechos selecionados, marcados com `inferred: true`.

`status` distingue `answered`, `refused_no_context` (nenhum trecho passou nos filtros de retrieval) e `refused_by_model` (o gerador recusou ou a resposta não passou na verificação). Toda recusa devolve o mesmo texto canônico, `sources: []` e `confidence: "baixa"`; `refusal_reason` explica o motivo: `declared` (o modelo declarou não ter sustentação), `pattern` (formulação de recusa reconhecida), `unsupported` (resposta com pouca sobreposição com o contexto), `unsupported_numbers` (prazo/valor/percentual ausente das fontes) ou `no_context`.

A pergunta é normalizada (`strip`) e precisa ter de 2 a 2000 caracteres; caracteres de controle (exceto quebra de linha e tab) são rejeitados com `422`.

### Erros

Toda resposta de erro tem o mesmo formato e nunca inclui detalhes internos (URL do Ollama, mensagem de exceção, stack trace) — esses ficam no log, correlacionados pelo `request_id`:

```json
{"detail": "Serviço de resposta temporariamente indisponível. Tente novamente em instantes.", "error_code": "provider_unavailable", "request_id": "3f2c…"}
```

| HTTP | `error_code` | Quando |
|---|---|---|
| 422 | `invalid_question` (ou erro de validação do Pydantic) | pergunta vazia, curta demais, longa demais ou com caracteres de controle |
| 503 | `index_not_ready` | o índice ainda não foi construído (requisição antes do fim do boot) |
| 503 | `provider_unavailable` | Ollama recusou conexão ou devolveu HTTP de erro (ex.: modelo não instalado) |
| 503 | `provider_timeout` | Ollama excedeu `OLLAMA_EMBED_TIMEOUT_S` / `OLLAMA_GENERATE_TIMEOUT_S` |
| 503 | `provider_invalid_response` | Ollama respondeu algo fora do contrato (JSON inválido, resposta vazia, nº de embeddings errado) |
| 500 | `internal_error` | falha inesperada; o stack trace está no log com o mesmo `request_id` |
| 401 | `unauthorized` | `API_TOKEN` configurado e credencial ausente/inválida |
| 429 | `rate_limited` | limite por IP excedido (`Retry-After` informa quando tentar de novo) |

### Cache e concorrência

Respostas são cacheadas em memória (LRU + TTL; `RAG_CACHE_MAX_ENTRIES`, padrão 256, `0` desliga; `RAG_CACHE_TTL_S`, padrão 600 s). A chave ignora caixa, acentos e pontuação e inclui a versão do índice e do prompt — reindexar ou mudar o template invalida tudo. Só respostas e recusas por falta de contexto entram; recusas do modelo não. Um acerto devolve `timings_ms: {"cache": 0}` e um `request_id` novo, e é logado como `query.cached`.

Chamadas ao Ollama passam por um semáforo (`OLLAMA_MAX_CONCURRENCY`, padrão 2): em CPU, requisições além disso esperam na fila em vez de estourar timeout; o tempo de espera aparece como `queue_wait_ms` nos eventos `provider.*`. Cada provider mantém uma única conexão HTTP com keep-alive, fechada no shutdown.

### Segurança (opcional)

A API é tratada como não pública (decisão D5), então nada disto está ligado por padrão:

| Variável | Efeito |
|---|---|
| `API_TOKEN` (≥ 16 caracteres) | exige `Authorization: Bearer <token>` em `/api/*` (`401` + `WWW-Authenticate` caso contrário); `/health` e `/ready` continuam livres |
| `RAG_RATE_LIMIT_PER_MINUTE` (> 0) | token bucket por IP em `POST /api/ask`; excesso → `429` com `Retry-After` e `error_code: rate_limited`. `RAG_RATE_LIMIT_BURST` define a capacidade (padrão = limite/min). Por processo: com N workers o limite efetivo é N× |
| `RAG_TRUST_PROXY=true` | usa `X-Forwarded-For` como IP do cliente (só atrás de proxy confiável; sem isso o cabeçalho é ignorado) |
| `RAG_DOCS_ENABLED=false` | desliga `/docs`, `/redoc` e `/openapi.json` |

### Saúde e prontidão

- `GET /health` — **liveness**: o processo responde. Nunca consulta o Ollama. Inclui `chunks` e `mode` quando o índice existe (compatibilidade).
- `GET /ready` — **readiness**: `200` só quando o índice tem chunks e, em `RAG_MODE=ollama`, o servidor responde em `/api/tags` com os dois modelos configurados instalados. Caso contrário `503` com `checks` detalhando o que falta (`missing_models`, `error_code`). Use este endpoint em orquestradores para só rotear tráfego quando o serviço puder responder.

### Inicialização e índice persistido

O índice é construído **uma única vez** no `lifespan` do FastAPI, antes de o servidor aceitar conexões, e **persistido** em `RAG_INDEX_DIR` (padrão `.rag_index/`, ignorado pelo Git): `vectors.npy` (matriz normalizada), `chunks.json` e `manifest.json` (modelo e dimensão dos embeddings, versão e parâmetros do chunking, prefixos de tarefa, `sha256` de cada arquivo do corpus, nº de chunks, data). No boot seguinte o manifesto é comparado com o estado atual: se corpus, modelo e chunking não mudaram, o índice é carregado do disco sem chamar o Ollama; qualquer diferença (arquivo alterado/novo/removido, troca de `OLLAMA_EMBED_MODEL`, prefixos, chunking) reconstrói e regrava, com o motivo no evento `index.rebuilt`. `RAG_INDEX_DIR=` (vazio) desliga a persistência.

```bash
python -m app.ingest            # (re)indexa a partir de corpus/ (ou CORPUS_DIR) — útil em CI/CD e no build da imagem
python -m app.ingest --check    # exit 1 se o índice persistido estiver desatualizado
python -m app.ingest --force    # reconstrói mesmo com manifesto compatível
```

`RAGService.reload()` reconstrói e troca o índice em uso sem reiniciar o processo (base para um endpoint administrativo protegido por token, opcional).

 Configuração inválida (`RAG_MODE` desconhecido, `RAG_TOP_K` fora de `1..50`, `RAG_MIN_SCORE` fora de `0..1`, URL do Ollama malformada, timeouts fora de `1..600 s`) ou índice impossível de construir (corpus vazio, Ollama inacessível) fazem o processo **encerrar no boot** com a variável problemática no log (`startup.failed`), em vez de servir `500`.

## Observabilidade

Toda requisição recebe um `X-Request-ID` (gerado, ou reaproveitado do cabeçalho enviado pelo cliente se tiver formato seguro) que volta no cabeçalho da resposta, no campo `request_id` do corpo e em todos os eventos de log daquela requisição. Os logs são estruturados (uma linha JSON por evento; `LOG_FORMAT=text` para leitura humana) e emitidos em stdout:

| Evento | Quando | Campos principais |
|---|---|---|
| `index.built` / `index.error` | ao construir o índice | documentos, chunks, dimensão, modelos, duração |
| `query.retrieved` | após retrieval + filtros | `candidates` (ids e scores do top-k), `selected` |
| `query.answered` | ao final de cada pergunta | `status` (`answered`, `refused_no_context`, `refused_by_model`), confiança, nº de fontes, `refusal_reason`, `support`, `timings_ms`, `total_ms` |
| `answer.refused` | quando o gerador respondeu mas a resposta foi recusada | motivo, sustentação medida, se o modelo declarou `grounded`, padrão casado, números não sustentados |
| `provider.embed` / `provider.generate` | a cada chamada ao Ollama | tokens, `done_reason`, `prompt_version`, se a saída veio estruturada, `queue_wait_ms`, durações reportadas pelo servidor |
| `query.cached` | resposta servida do cache | status, nº de fontes, estatísticas do cache (hits, misses, hit_rate) |
| `provider.error` | falha em qualquer etapa | etapa, tipo do erro, stack trace |
| `http.error` | resposta de erro da API | status, `error_code`, tipo e mensagem interna do erro (o cliente recebe só o genérico) |
| `http.request` | a cada requisição HTTP | método, path, status, duração (`/health` e `/ready` só em DEBUG) |
| `settings.loaded` / `startup.failed` | no boot | configuração efetiva (sem credenciais) e origem de cada valor; motivo da falha de boot |
| `index.loaded` / `index.rebuilt` / `index.reloaded` | no boot ou em `reload()` | índice carregado do disco, ou reconstruído com o motivo (`reason`), ou recarregado sob demanda |
| `ingest.file` / `ingest.skipped` / `ingest.duplicate` / `ingest.empty` | na ingestão | diagnóstico por arquivo (páginas, linhas, delimitador, hash, chunks), formatos ignorados, duplicatas descartadas |
| `ready.ollama_unreachable` | em `/ready`, modo Ollama | `error_code` da sonda ao Ollama |

Pergunta e resposta em texto integral (`query.text`) só são registradas em `LOG_LEVEL=DEBUG`, porque perguntas podem conter dados pessoais.

## Testes e evals

```bash
pytest -q                      # suíte (configuração em pyproject.toml; não precisa de PYTHONPATH)
coverage run -m pytest -q && coverage report
ruff check app evals tests && ruff format --check app evals tests
```

A suíte (380+ testes) cobre ingestão de PDF real gerado em teste, CSV, Markdown e texto; chunking (invariantes em textos aleatórios); embeddings e contrato do Ollama simulado (`httpx.MockTransport`: lotes, retry, timeout, 404 de modelo, dimensão inconsistente, `<think>`, `done_reason=length`, JSON inválido); retrieval híbrido e limiares; recusa e sustentação; fontes; cache e concorrência; persistência do índice; ciclo de vida da API, `/ready` e contrato de erro; o harness de avaliação; e a coerência entre `pyproject.toml`, `uv.lock` e os `requirements*.txt`. Piso de cobertura na CI: 85 % (medido: 96 %). Os mutantes que sobreviviam na auditoria (remover `min_score`, confiança sempre alta, citar uma fonte, remover overlap, girar o ranking, 503→500, recusa por prefixo, canal lexical, MMR) são mortos pela suíte; o único que sobrevive — remover o sinal do hash local — é uma limitação documentada do provider de teste (R-07). A CI roda em Python 3.11, 3.12 e 3.13 e falha em: `ruff check`/`ruff format --check`, `mypy --strict` sobre `app/` e `evals/`, cobertura < 85 %, gate do eval local (`tests/test_evals.py`), lockfile desatualizado ou `requirements*.txt` divergentes, vulnerabilidades conhecidas (`pip-audit`), whitespace/marcadores de conflito no diff do PR, e no job `docker`, que constrói a imagem, sobe o contêiner, sonda `/ready` e faz uma pergunta real.

### Avaliação do RAG (evals)

Os casos ficam em [`evals/cases.json`](evals/cases.json) e são executados sobre o **corpus real** em `corpus/`. Cada caso tem categoria (`in_scope`, `partial`, `out_of_scope`, `typo`, `no_accent`, `synonym`, `adversarial`), documentos/chunks esperados, fragmentos obrigatórios e proibidos na resposta e se a resposta deve ou não citar fontes.

```bash
python -m evals.run                      # modo local (hash + extrativo): retrieval e recusa, sem LLM
python -m evals.run --mode ollama --save # modo principal; grava evals/results/<timestamp>-ollama.json
python -m evals.run --show-failures      # mostra resposta, candidatos e fontes dos casos reprovados
python -m evals.run --k 8 --min-score 0.3
```

Métricas reportadas (definições em `evals/harness.py`): `recall@k` e `MRR` dos candidatos do índice, `selected recall` (o que chega ao gerador), `source precision` (fontes citadas que pertencem aos documentos esperados), `correct refusal` (fora de escopo recusado sem fontes), `false refusal` (recusa indevida), `content pass` (verificações de conteúdo) e latência p50/p95, no total e por categoria.

A busca vetorial roda sobre uma matriz `numpy` normalizada (`app/store.py`, interface `VectorStore` plugável): 10 mil chunks × 768 dimensões respondem em < 1 ms, e um filtro por documento/seção/metadado restringe a busca sem reindexar.

O retrieval é **híbrido**: cosseno sobre embeddings e BM25 sobre texto normalizado (sem acentos, stopwords PT-BR, radicais), fundidos por RRF; os filtros de evidência rodam sobre o pool fundido (4·k por canal) antes do corte em k.

#### Limiares por provider

A escala do cosseno depende do modelo de embedding, então os limiares vêm de um **perfil por modo** (`app/config.py: THRESHOLD_PROFILES`, espelhado em `evals/thresholds.json → profiles`): `min_score` (piso de cosseno), `vector_only_min_score` (aceita sem termo em comum), `vector_with_overlap_min_score` (aceita com um radical em comum), `min_lexical_coverage` (cobertura lexical ponderada por IDF), `high_confidence_score` e `relative_gap` (confiança). Qualquer um pode ser sobrescrito por variável de ambiente (`RAG_MIN_SCORE`, `RAG_VECTOR_ONLY_MIN_SCORE`, `RAG_VECTOR_WITH_OVERLAP_MIN_SCORE`, `RAG_MIN_LEXICAL_COVERAGE`, `RAG_HIGH_CONFIDENCE_SCORE`, `RAG_RELATIVE_GAP`).

Para calibrar um provider, `python -m evals.calibrate --mode ollama` executa o retrieval uma vez por caso e varre uma grade de limiares, imprimindo recusa correta × recusa indevida, selected recall e precisão de fontes para cada combinação (o perfil atual aparece destacado). O perfil `local` foi calibrado assim; o perfil `ollama` é **provisório** (derivado da escala típica de modelos densos) até ser medido com o modelo real.

Depois dos filtros, os trechos aprovados passam por **MMR** (diversificação: um quase-duplicado do top-1 não ocupa a vaga de um trecho diferente e relevante), controlado por `mmr_lambda` no perfil (`1.0` = desligado no perfil `local`, onde o cosseno do hash é ruidoso; `0.7` no perfil `ollama`; `RAG_MMR_LAMBDA` sobrescreve). A interface `Reranker` (`app/rerank.py`) permite plugar um reranker real via `RAG_RERANKER`; hoje só existe `noop` — um cross-encoder em CPU custaria 0,3–0,8 s por consulta e só entra se o eval mostrar ≥ 5 p.p. de MRR (decisão D3).

`confidence` combina três sinais: score do top-1 na escala do provider, destaque do top-1 sobre o top-2 (gap relativo) **ou** dois trechos do mesmo documento concordando, e a sustentação medida da resposta (`alta` exige as três; sustentação < 0,6 rebaixa para `baixa`).

O corpus fica em `corpus/` (`CORPUS_DIR` para apontar outro diretório); `docs/` contém a documentação do projeto. O corpus aceita `.pdf`, `.csv`, `.md` e `.txt` (outros formatos são ignorados com aviso). CSV é lido com a biblioteca padrão — delimitador detectado (`,` `;` tab `|`), BOM tolerado, tudo como texto (`00123` não vira `123`); cada linha é indexada com os nomes das colunas (para busca) mas exibida ao gerador e ao usuário só com o conteúdo. Cada arquivo gera um evento `ingest.file` (páginas, páginas sem texto, linhas, delimitador, chunks, hash); arquivo ilegível falha no boot citando o nome. Trechos idênticos ou quase idênticos (Jaccard de radicais ≥ 0,9) entre documentos são deduplicados, mantendo o primeiro na ordem alfabética dos arquivos.

Os PDFs são divididos por **seção** (títulos numerados como `2. Dados coletados`), com cabeçalho/rodapé repetidos removidos, quebras de linha visuais desfeitas e um orçamento de ~300 tokens por chunk (`app/chunking.py`); cada chunk carrega `section`, posição no texto e estimativa de tokens.

`tests/test_evals.py` roda os casos em modo local a cada `pytest` e compara com os pisos de [`evals/thresholds.json`](evals/thresholds.json) — a baseline medida em P0-03 vira gate de regressão e deve subir a cada melhoria do pipeline. O gate do modo Ollama (critérios de aceite do plano) só roda com `RAG_EVAL_OLLAMA=1 pytest -m ollama` e um servidor acessível em `OLLAMA_BASE_URL`; sem isso é pulado. Os relatórios em `evals/results/` não são versionados.

### Atualizar dependências

```bash
uv lock --upgrade                                   # ou edite os pisos em pyproject.toml e rode `uv lock`
uv export --no-dev --no-hashes --no-emit-project --format requirements-txt --output-file requirements.txt
uv export --only-dev --no-hashes --no-emit-project --format requirements-txt --output-file requirements-dev.txt
```

Os três arquivos devem ser commitados juntos; a CI falha se `requirements*.txt` divergirem de `uv.lock`.

## Decisões importantes

- **Sem respostas principais hardcoded.** A resposta nasce do contexto recuperado.
- **Lógica central visível em Python.** O projeto evita esconder retrieval e geração atrás de abstrações desnecessárias.
- **Providers substituíveis.** Embeddings e geração usam interfaces simples.
- **Transparência.** O modo offline é extrativo; somente o modo Ollama é descrito como generativo.
- **Falha explícita.** Erro de provider retorna `503` com `error_code` estável, não sucesso inventado; configuração inválida derruba o boot, não a primeira requisição.
- **Recusa antes de citação.** Perguntas sem contexto suficiente não recebem fontes arbitrárias. A recusa não depende de uma frase exata do modelo: o gerador Ollama responde em JSON (`answer`, `grounded`, `used_sources`) e, independentemente disso, a resposta passa por um classificador de recusa em PT-BR e por uma verificação de sustentação (tokens de conteúdo e quantidades precisam existir no contexto). Só respostas que passam recebem fontes.

## Stack

`Python` · `FastAPI` · `Pydantic` · `NumPy` · `PyPDF` · `HTTPX` · `Pytest` · `Docker` · `GitHub Actions` · `Ollama`

## Docker

```bash
docker compose up --build              # API + Ollama + pull dos modelos (modo ollama, 100 % CPU por padrão)
RAG_MODE=local docker compose up app   # só a API, modo local
docker build -t aurora-rag . && docker run -p 8000:8000 aurora-rag   # imagem isolada, modo local
```

A imagem (`python:3.12-slim`) instala apenas as dependências de runtime travadas pelo lockfile, copia só `app/` e `corpus/` (`.dockerignore` em lista branca — `.git`, `.env`, testes e auditoria ficam fora), roda como usuário não-root (`aurora`, uid 10001), pré-constrói o índice no build para o modo local (boot em ~20 ms) e declara `HEALTHCHECK` em `/health`. O índice persistido fica no volume `/data/index`; no modo ollama ele é (re)construído no primeiro boot contra o servidor e reaproveitado nos seguintes. O `docker-compose.yml` sobe `ollama` com volume de modelos, um serviço `ollama-pull` que baixa os dois modelos uma única vez, e a API só inicia depois disso; a readiness do compose usa `/ready`. Para GPU NVIDIA, descomente o bloco `deploy.resources` do serviço `ollama`.

## Deploy

O repositório mantém configuração de container/deploy. Uma URL pública de backend só será anunciada quando a aplicação estiver realmente implantada e monitorada. A interface em `/` (`app/static/`) é um cliente fino da própria API — mostra status, confiança, fontes com trecho e `request_id` — e não existe demo estática separada.

## Uso de IA no desenvolvimento

Ferramentas de IA podem acelerar implementação, revisão e documentação. O que apresento como evidência é auditável no próprio repositório: arquitetura, código, testes, evals e limitações explícitas.
