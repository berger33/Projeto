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
| arquitetura | [`ARQUITETURA.md`](ARQUITETURA.md) |

## Arquitetura

```text
PDF / CSV
   ↓
Document loader + chunking
   ↓
Embedding provider
   ├─ local hash embedding (CI/offline)
   └─ Ollama / nomic-embed-text
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

Abra `http://127.0.0.1:8000` ou `/docs`.

### RAG generativo com Ollama

```bash
ollama pull nomic-embed-text
ollama pull qwen3:0.6b
```

Configure:

```env
RAG_MODE=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:0.6b
OLLAMA_EMBED_TIMEOUT_S=30        # opcional (1..600)
OLLAMA_GENERATE_TIMEOUT_S=60     # opcional (1..600); em CPU, modelos maiores podem precisar de mais
```

`GET /ready` confirma se o servidor e os dois modelos estão disponíveis antes de enviar perguntas.

## API

`POST /api/ask`

```json
{"question":"Qual é o prazo para devolução?"}
```

A resposta contém `answer`, `sources`, `confidence`, `mode`, `request_id` e `timings_ms` (ms por etapa: `retrieve`, `filter`, `generate`). Fontes são derivadas dos chunks realmente selecionados e preservam documento/página/linha quando disponíveis.

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

### Saúde e prontidão

- `GET /health` — **liveness**: o processo responde. Nunca consulta o Ollama. Inclui `chunks` e `mode` quando o índice existe (compatibilidade).
- `GET /ready` — **readiness**: `200` só quando o índice tem chunks e, em `RAG_MODE=ollama`, o servidor responde em `/api/tags` com os dois modelos configurados instalados. Caso contrário `503` com `checks` detalhando o que falta (`missing_models`, `error_code`). Use este endpoint em orquestradores para só rotear tráfego quando o serviço puder responder.

### Inicialização

O índice é construído **uma única vez** no `lifespan` do FastAPI, antes de o servidor aceitar conexões. Configuração inválida (`RAG_MODE` desconhecido, `RAG_TOP_K` fora de `1..50`, `RAG_MIN_SCORE` fora de `0..1`, URL do Ollama malformada, timeouts fora de `1..600 s`) ou índice impossível de construir (corpus vazio, Ollama inacessível) fazem o processo **encerrar no boot** com a variável problemática no log (`startup.failed`), em vez de servir `500`.

## Observabilidade

Toda requisição recebe um `X-Request-ID` (gerado, ou reaproveitado do cabeçalho enviado pelo cliente se tiver formato seguro) que volta no cabeçalho da resposta, no campo `request_id` do corpo e em todos os eventos de log daquela requisição. Os logs são estruturados (uma linha JSON por evento; `LOG_FORMAT=text` para leitura humana) e emitidos em stdout:

| Evento | Quando | Campos principais |
|---|---|---|
| `index.built` / `index.error` | ao construir o índice | documentos, chunks, dimensão, modelos, duração |
| `query.retrieved` | após retrieval + filtros | `candidates` (ids e scores do top-k), `selected` |
| `query.answered` | ao final de cada pergunta | `status` (`answered`, `refused_no_context`, `refused_by_model`), confiança, nº de fontes, `timings_ms`, `total_ms` |
| `provider.embed` / `provider.generate` | a cada chamada ao Ollama | tokens, `done_reason`, durações reportadas pelo servidor |
| `provider.error` | falha em qualquer etapa | etapa, tipo do erro, stack trace |
| `http.error` | resposta de erro da API | status, `error_code`, tipo e mensagem interna do erro (o cliente recebe só o genérico) |
| `http.request` | a cada requisição HTTP | método, path, status, duração (`/health` e `/ready` só em DEBUG) |
| `settings.loaded` / `startup.failed` | no boot | configuração efetiva (sem credenciais) e origem de cada valor; motivo da falha de boot |
| `ready.ollama_unreachable` | em `/ready`, modo Ollama | `error_code` da sonda ao Ollama |

Pergunta e resposta em texto integral (`query.text`) só são registradas em `LOG_LEVEL=DEBUG`, porque perguntas podem conter dados pessoais.

## Testes e evals

```bash
pytest -q                      # suíte (configuração em pyproject.toml; não precisa de PYTHONPATH)
coverage run -m pytest -q && coverage report
ruff check app evals tests && ruff format --check app evals tests
```

A suíte cobre ingestão CSV, chunking, retrieval, recusa fora da base, rastreabilidade das fontes, validação de entrada, contrato HTTP, o harness de avaliação e a coerência entre `pyproject.toml`, `uv.lock` e os `requirements*.txt`. A CI roda em Python 3.11, 3.12 e 3.13, verifica lint/formatação, cobertura mínima, atualidade do lockfile e vulnerabilidades conhecidas (`pip-audit`).

### Avaliação do RAG (evals)

Os casos ficam em [`evals/cases.json`](evals/cases.json) e são executados sobre o **corpus real** em `docs/`. Cada caso tem categoria (`in_scope`, `partial`, `out_of_scope`, `typo`, `no_accent`, `synonym`, `adversarial`), documentos/chunks esperados, fragmentos obrigatórios e proibidos na resposta e se a resposta deve ou não citar fontes.

```bash
python -m evals.run                      # modo local (hash + extrativo): retrieval e recusa, sem LLM
python -m evals.run --mode ollama --save # modo principal; grava evals/results/<timestamp>-ollama.json
python -m evals.run --show-failures      # mostra resposta, candidatos e fontes dos casos reprovados
python -m evals.run --k 8 --min-score 0.3
```

Métricas reportadas (definições em `evals/harness.py`): `recall@k` e `MRR` dos candidatos do índice, `selected recall` (o que chega ao gerador), `source precision` (fontes citadas que pertencem aos documentos esperados), `correct refusal` (fora de escopo recusado sem fontes), `false refusal` (recusa indevida), `content pass` (verificações de conteúdo) e latência p50/p95, no total e por categoria.

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
- **Recusa antes de citação.** Perguntas sem contexto suficiente não recebem fontes arbitrárias.

## Stack

`Python` · `FastAPI` · `Pydantic` · `Pandas` · `PyPDF` · `HTTPX` · `Pytest` · `Docker` · `GitHub Actions` · `Ollama`

## Deploy

O repositório mantém configuração de container/deploy. Uma URL pública de backend só será anunciada quando a aplicação estiver realmente implantada e monitorada; a demo HTML não é apresentada como substituta do backend.

## Uso de IA no desenvolvimento

Ferramentas de IA podem acelerar implementação, revisão e documentação. O que apresento como evidência é auditável no próprio repositório: arquitetura, código, testes, evals e limitações explícitas.
