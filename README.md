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
```

## API

`POST /api/ask`

```json
{"question":"Qual é o prazo para devolução?"}
```

A resposta contém `answer`, `sources`, `confidence`, `mode`, `request_id` e `timings_ms` (ms por etapa: `retrieve`, `filter`, `generate`). Fontes são derivadas dos chunks realmente selecionados e preservam documento/página/linha quando disponíveis.

`GET /health` informa saúde, quantidade de chunks indexados e modo ativo.

## Observabilidade

Toda requisição recebe um `X-Request-ID` (gerado, ou reaproveitado do cabeçalho enviado pelo cliente se tiver formato seguro) que volta no cabeçalho da resposta, no campo `request_id` do corpo e em todos os eventos de log daquela requisição. Os logs são estruturados (uma linha JSON por evento; `LOG_FORMAT=text` para leitura humana) e emitidos em stdout:

| Evento | Quando | Campos principais |
|---|---|---|
| `index.built` / `index.error` | ao construir o índice | documentos, chunks, dimensão, modelos, duração |
| `query.retrieved` | após retrieval + filtros | `candidates` (ids e scores do top-k), `selected` |
| `query.answered` | ao final de cada pergunta | `status` (`answered`, `refused_no_context`, `refused_by_model`), confiança, nº de fontes, `timings_ms`, `total_ms` |
| `provider.embed` / `provider.generate` | a cada chamada ao Ollama | tokens, `done_reason`, durações reportadas pelo servidor |
| `provider.error` | falha em qualquer etapa | etapa, tipo do erro, stack trace |
| `http.request` | a cada requisição HTTP | método, path, status, duração (`/health` só em DEBUG) |

Pergunta e resposta em texto integral (`query.text`) só são registradas em `LOG_LEVEL=DEBUG`, porque perguntas podem conter dados pessoais.

## Testes e evals

```bash
pytest -q                      # suíte (configuração em pyproject.toml; não precisa de PYTHONPATH)
coverage run -m pytest -q && coverage report
ruff check app tests && ruff format --check app tests
```

A suíte cobre ingestão CSV, chunking, retrieval, recusa fora da base, rastreabilidade das fontes, validação de entrada, contrato HTTP, casos de comportamento versionados e a coerência entre `pyproject.toml`, `uv.lock` e os `requirements*.txt`. A CI roda em Python 3.11, 3.12 e 3.13, verifica lint/formatação, cobertura mínima, atualidade do lockfile e vulnerabilidades conhecidas (`pip-audit`).

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
- **Falha explícita.** Erro de provider retorna `503`, não sucesso inventado.
- **Recusa antes de citação.** Perguntas sem contexto suficiente não recebem fontes arbitrárias.

## Stack

`Python` · `FastAPI` · `Pydantic` · `Pandas` · `PyPDF` · `HTTPX` · `Pytest` · `Docker` · `GitHub Actions` · `Ollama`

## Deploy

O repositório mantém configuração de container/deploy. Uma URL pública de backend só será anunciada quando a aplicação estiver realmente implantada e monitorada; a demo HTML não é apresentada como substituta do backend.

## Uso de IA no desenvolvimento

Ferramentas de IA podem acelerar implementação, revisão e documentação. O que apresento como evidência é auditável no próprio repositório: arquitetura, código, testes, evals e limitações explícitas.
