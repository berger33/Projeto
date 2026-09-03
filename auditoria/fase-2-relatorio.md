# Fase 2 — Auditoria Técnica Completa

**Repositório:** `berger33/aurora-document-rag` · **Commit base:** `1183cf4` · **Data:** 2026-09-02
**Escopo:** todo o repositório (2.1) + pipeline RAG em profundidade (2.2)
**Método:** leitura integral do código + execução instrumentada na sandbox (Python 3.11, deps resolvidas hoje) + simulação do caminho Ollama com `httpx.MockTransport` (não há servidor Ollama na sandbox) + `pip-audit`, `ruff`, `coverage` e teste de mutação manual. **Nenhum arquivo de código foi alterado**; todos os mutantes foram revertidos e verificados com `git status`.
**Decisões já tomadas (Fase 1):** alvo principal = modo `ollama` local, sem provedores pagos; modo `local` = harness de teste; relatórios em `auditoria/`.

Severidade: **Crítico** = resposta errada/vazamento/dado corrompido no caminho principal · **Alto** = degrada qualidade ou disponibilidade de forma provável · **Médio** = robustez/manutenibilidade/risco condicional · **Baixo** = cosmético/documental.

## Sumário executivo

| | Crítico | Alto | Médio | Baixo | Total |
|---|---|---|---|---|---|
| 2.1 Geral (G-01…G-30) | 0 | 3 | 15 | 12 | 30 |
| 2.2 RAG (R-01…R-25) | 2 | 11 | 7 | 5 | 25 |
| **Total** | **2** | **14** | **22** | **17** | **55** |

**Os cinco achados que mais importam para o objetivo (RAG como ponto forte, modo Ollama):**

1. **R-05 (Crítico)** — `nomic-embed-text` é usado sem os prefixos obrigatórios `search_document:`/`search_query:` e é um modelo treinado para inglês; o corpus é PT-BR. O retrieval semântico do caminho principal opera abaixo do que o próprio modelo entrega.
2. **R-13 (Crítico)** — Recusa do LLM é detectada por `startswith("não encontrei informação suficiente")`. 5 de 7 formulações realistas de recusa passam despercebidas e recebem **fontes e confiança "média"** — exatamente o cenário que o README promete evitar ("recusa antes de citação").
3. **R-03 (Alto)** — `split_text` tem três defeitos: parágrafo maior que `chunk_size` após buffer não vazio **nunca é fatiado** (observado chunk de 5.008 chars com limite 900); `tail+parágrafo` pode exceder o limite; o texto do `pypdf` não contém `\n\n`, então a estratégia "por parágrafo" nunca dispara e todo PDF é cortado por caractere no meio de palavras.
4. **R-11 / R-14 (Alto)** — `min_score=0.12` e o limiar de confiança `0.45` foram calibrados implicitamente para o hash embedding; com cosseno de `nomic` (relevantes ≈0,6–0,8; irrelevantes ≈0,3–0,5) o `min_score` não filtra nada e quase tudo vira "alta". Não há orçamento de contexto nem `num_ctx`: prompt grande é truncado silenciosamente pelo Ollama.
5. **R-20 / G-20 / R-22 (Alto)** — Zero `logging`; testes não cobrem PDF, Ollama nem o caminho 503; 7 de 8 mutantes sobrevivem à suíte (remover o filtro `min_score`, inverter o ranking e fixar confiança em "alta" **não quebram nenhum teste**). Somam-se G-01/G-02 (Alto): falha de boot vira 500 genérico e o 503 expõe mensagem interna.

Não foram encontrados segredos expostos nem vulnerabilidades conhecidas nas versões **resolvidas hoje** — mas as faixas do `requirements.txt` admitem versões com CVEs (G-11).

---

## 2.1 Auditoria geral do repositório

### 2.1.1 Bugs de lógica e tratamento de erros

#### G-01 · Alto · `app/main.py:33-50`, `app/rag.py:21-30` — Falha na inicialização vira HTTP 500 sem tratamento; `/health` mente ou explode
- **Descrição:** `RAGService` é construído lazily na primeira requisição via `@lru_cache`. Se o Ollama estiver fora do ar, o modelo não existir ou `RAG_MODE` for inválido, a exceção sobe do `Depends` sem `try/except` → `500 Internal Server Error` genérico, inclusive em `/health`. Quando sobe, `/health` devolve `"ok"` sem sondar o Ollama.
- **Evidência:** `RAG_MODE=ollama OLLAMA_BASE_URL=http://127.0.0.1:9` → `GET /health` = 500; `RAG_MODE=openai` → `create_app()` sucede, `/health` = 500.
- **Impacto:** orquestradores (Render `healthCheckPath: /health`, Docker) não distinguem "subindo", "config inválida" e "Ollama indisponível"; deploy em loop de restart sem mensagem útil.
- **Recomendação:** validar `Settings` no import/`lifespan`; construir o serviço no `lifespan` do FastAPI; `/health` (liveness) separado de `/ready` (readiness com ping ao Ollama e checagem de `chunks > 0`); handler global para erros de provider.

#### G-02 · Alto · `app/main.py:58-59` — Mensagem interna da exceção é exposta ao cliente
- **Descrição:** `except Exception as exc: raise HTTPException(503, detail=f"RAG indisponível: {exc}")`.
- **Evidência:** resposta real observada: `{"detail":"RAG indisponível: [Errno 111] Connection refused to http://10.0.0.5:11434/api/generate (token=abc)"}` — URL interna e qualquer conteúdo da mensagem vazam.
- **Impacto:** divulgação de topologia interna (host/porta do Ollama, nomes de modelo, caminhos). Também dificulta contrato estável de erro.
- **Recomendação:** `detail` genérico + `error_code` estável (`provider_unavailable`, `provider_timeout`, `index_not_ready`); mensagem completa só no log com `request_id`.

#### G-03 · Médio · `app/main.py:33-35` — `lru_cache` não serializa a construção concorrente
- **Descrição:** endpoints síncronos rodam no threadpool do Starlette; N primeiras requisições simultâneas constroem N `RAGService` (N indexações completas no Ollama) antes de o cache ser preenchido.
- **Evidência:** simulação com 8 threads → 8 construções.
- **Recomendação:** construir no `lifespan` (resolve também G-01) ou proteger com `threading.Lock`.

#### G-04 · Médio · `app/config.py:26-27` — Variáveis de ambiente malformadas ou fora de faixa não são validadas
- **Evidência:** `RAG_TOP_K=abc` → `ValueError` só na primeira request (→500 via G-01); `RAG_MIN_SCORE=-1` é aceito (desliga o filtro); não há limite superior para `RAG_TOP_K`; timeouts dos providers não são configuráveis.
- **Recomendação:** `pydantic-settings` (ou validação explícita) com faixas (`0 ≤ min_score ≤ 1`, `1 ≤ top_k ≤ 50`), mensagens claras e falha no boot.

#### G-05 · Médio · `app/documents.py:58-61` — Inferência de dtype do `pandas` corrompe valores textuais do CSV
- **Evidência:** `codigo=00123` → `123`; `valor=10.50` → `10.5`; célula vazia vira `""` (ok). CSV com `;` (padrão Excel PT-BR) e BOM vira **uma única coluna** `"pergunta;resposta: Qual prazo?;10 dias"`.
- **Impacto:** CEPs, códigos de pedido, SKUs e valores monetários indexados com texto diferente do documento; CSV `;` indexado como lixo sem erro.
- **Recomendação:** `pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python", encoding="utf-8-sig")` ou `csv.DictReader` com `Sniffer`. Cobrir com teste.

#### G-06 · Médio · `app/documents.py:47-64` — Ingestão silenciosa e sem diagnóstico
- **Evidência:** `.txt/.md/.docx/.PDF-corrompido` → ignorados ou `PdfStreamError` bruto; CSV vazio → `EmptyDataError` do pandas; diretório inexistente → mensagem genérica "Nenhum conteúdo PDF/CSV encontrado"; página de PDF sem texto (escaneada) gera zero chunks sem aviso; subpastas ignoradas.
- **Recomendação:** registrar por arquivo (formato, páginas, chunks, páginas vazias); erro por arquivo com nome; extensão case-insensitive já ok; decidir política para formatos não suportados (aviso vs erro).

#### G-07 · Médio · `app/embeddings.py:52-55`, `app/retrieval.py:9-11` — Dimensão dos embeddings não é validada na indexação
- **Evidência:** provider devolvendo vetores `[2 dims, 3 dims]` é aceito; o erro só aparece por consulta em `cosine_similarity` → 503 em **todas** as requests.
- **Recomendação:** validar dimensão única no `VectorIndex.__init__` e falhar no boot com mensagem clara (modelo trocado sem reindexar é o caso real).

#### G-08 · Médio · `app/retrieval.py:26`, `app/embeddings.py:40,45-51` — Indexação em uma única requisição HTTP com timeout de 30 s, sem batching nem retry
- **Descrição:** todos os textos do corpus são enviados numa única chamada `/api/embed`. Em CPU, centenas de chunks facilmente excedem 30 s → boot falha (→ G-01).
- **Evidência:** `embed(500 textos)` → 1 request com 500 itens.
- **Recomendação:** lotes (ex.: 32–64), timeout proporcional, retry com backoff para erros transitórios, reuso de `httpx.Client`.

#### G-09 · Baixo · `app/main.py:70` — UI inline mostra `\n` literal e `[object Object]`
- **Evidência:** a string Python contém `'\\\\n\\\\nFontes: '`, que chega ao navegador como `\\n\\nFontes:` → JS renderiza barra + "n" (verificado com Node: `includes('\n') === false`). Em 422 do Pydantic, `d.detail` é lista → `textContent = "[object Object]"`. Erro de rede deixa "Consultando..." travado.
- **Recomendação:** mover HTML/JS para arquivo estático (`StaticFiles`/`Jinja2`) e corrigir escapes; tratar `detail` array.

#### G-10 · Baixo · `app/main.py:17`, `app/rag.py:37-39` — Validação de entrada duplicada e incompleta
- **Evidência:** `"  "` (2 espaços) passa no Pydantic e cai no `ValueError` do domínio (422 com outra mensagem); `"\x00\x00\x00"` é aceito; não há `strip` no modelo.
- **Recomendação:** `field_validator` com `strip`, rejeição de controle/NUL, mensagem única.

### 2.1.2 Segredos, validação de inputs, vulnerabilidades

- **Segredos:** nenhum segredo encontrado no código, histórico (1 commit) ou `.env.example`; `.env` está no `.gitignore`. ✅
- **Vulnerabilidades em dependências resolvidas hoje:** `pip-audit -r requirements.txt` → nenhuma conhecida. ✅ (ver G-11 para o risco das faixas)

#### G-11 · Médio · `requirements.txt` — Sem lockfile; faixas admitem versões com CVEs; `pytest` em runtime
- **Evidência (`pip-audit` nas versões mínimas permitidas):** `pypdf 5.1.0` → GHSA-jm82-fx9c-mx94, CVE-2026-84309/84310/84311, CVE-2026-82398 (corrigidas ≥6.16.1); `starlette 0.41.3` (via `fastapi 0.115.6`) → PYSEC-2026-161/248/249/1941/1942/2280/2281; `pytest 8.3.0` → PYSEC-2026-1845. Um `pip install` com cache antigo ou resolução diferente pode instalar essas versões. `pytest` vai para a imagem Docker.
- **Recomendação:** `pyproject.toml` + lockfile (`uv lock`/`pip-compile`), grupos `dev`/`runtime`, elevar mínimos (`pypdf>=6.16.1`, `fastapi>=0.120`), `pip-audit` na CI.

#### G-12 · Médio · `Dockerfile`, ausência de `.dockerignore` — Imagem roda como root e copia tudo
- **Evidência:** `COPY . .` sem `.dockerignore` inclui `.git`, `.venv` (se existir), `tests`, `auditoria/` e **um `.env` local se existir** (segredo embarcado na imagem). Sem `USER`, sem `HEALTHCHECK`, sem `PYTHONDONTWRITEBYTECODE`/`PYTHONUNBUFFERED`.
- **Recomendação:** `.dockerignore`, usuário não-root, `HEALTHCHECK` em `/health`, copiar apenas `app/` e corpus.

#### G-13 · Médio · `app/main.py` — Sem autenticação, rate limit, limite de concorrência ou CORS explícito
- **Descrição:** `/api/ask` é público; cada chamada em modo `ollama` custa segundos de CPU/GPU no LLM. Sem limite, um único cliente satura o Ollama (DoS trivial). CORS: nenhum middleware (padrão = same-origin; ok para a UI embutida, mas indocumentado). `/docs` aberto.
- **Recomendação:** depende de decisão (§2.4-D5): ao menos rate limit por IP e semáforo de concorrência para o provider; token opcional via env para exposição pública.

#### G-14 · Médio · `app/generation.py:16-35` — Defesa contra prompt injection é apenas instrucional
- **Evidência:** um chunk contendo `RESPOSTA\n...\nPERGUNTA` reproduz os próprios marcadores do template dentro do contexto (verificado); `/api/generate` não separa system/user; sem sanitização dos delimitadores.
- **Recomendação:** `/api/chat` com `system` separado, delimitadores não triviais (ex.: tags XML com escape de `<`), e teste adversarial no eval.

### 2.1.3 Duplicação e complexidade

#### G-15 · Médio · três tokenizadores e três cópias da string de recusa
- `[a-zA-ZÀ-ÿ0-9]+` definido em `embeddings.py:15`, `generation.py:67`, `rag.py:17` (com filtros diferentes: `len>2` em dois deles, stopwords só em um). Mensagem "Não encontrei informação suficiente…" em `generation.py:86`, `rag.py:45` e comparada por prefixo em `rag.py:51`. `rstrip("/")` em `config.py:23`, `embeddings.py:41`, `generation.py:42`. Criação de `httpx.Client` duplicada.
- **Impacto:** divergência silenciosa (ex.: mudar a mensagem no gerador quebra a detecção no serviço).
- **Recomendação:** módulo `text.py` (normalização/tokenização) e constante/enum de recusa retornada de forma estruturada (não por string).

#### G-16 · Médio · `app/documents.py:16-44` — `split_text` é a função mais complexa e a mais defeituosa
- Complexidade ciclomática ≈8, três caminhos, três bugs (R-03), nenhum teste de borda. Recomendação: reescrever com estratégia explícita (ver 2.2) e testes de propriedade (tamanho máximo, cobertura total do texto, overlap efetivo, sem cortes intra-palavra).

#### G-17 · Baixo · `app/main.py:70` — 1 linha de 2.100 caracteres de HTML/CSS/JS dentro do Python (manutenibilidade; `ruff`/`black` não conseguem formatar).

### 2.1.4 Dependências (detalhe)

| Pacote | Declarado | Resolvido | Observação |
|---|---|---|---|
| fastapi | `>=0.115.6,<1.0` | 0.141.1 | mínimo arrasta starlette vulnerável (G-11) |
| starlette (transitiva) | — | 1.6.0 | emite `StarletteDeprecationWarning`: TestClient com `httpx` está deprecado ("install `httpx2`") → a suíte de testes vai quebrar numa versão futura sem pin (**G-18 · Médio**) |
| pypdf | `>=5.1,<7.0` | 6.16.2 | mínimo com 5 advisories |
| pandas | `>=2.2,<4.0` | 3.0.5 | usado só para `read_csv`; arrasta `numpy` (não usado) — substituível por `csv` stdlib (**G-19 · Baixo**) |
| httpx | `>=0.27,<1.0` | 0.28.1 | ok |
| uvicorn[standard] | `>=0.34,<1.0` | 0.52.4 | ok |
| pytest | `>=8.3,<10` | 9.1.1 | dependência de dev em runtime |
| Ollama (externo) | não fixado | — | nem versão do servidor nem digest dos modelos (`nomic-embed-text`, `qwen3:0.6b`) são fixados; `/health` não reporta |

### 2.1.5 Testes automatizados

#### G-20 · Alto · `tests/` — Cobertura nominal 75 %, cobertura efetiva baixa; suíte não detecta regressões no núcleo
- **Cobertura de linhas (`coverage --branch`):** `embeddings.py` 62 % (Ollama 0 %), `generation.py` 62 % (Ollama e `build_prompt` 0 %), `config.py` 68 % (`from_env` 0 %), `documents.py` 70 % (PDF 0 %), `retrieval.py` 75 %.
- **Nunca testados:** ingestão de PDF; `OllamaEmbeddingProvider`/`OllamaGenerator` (nem com mock); caminho 503; `build_prompt`; `Settings.from_env`; corpus real em `docs/`; bordas do chunking (H2/H3 acima); dedup de fontes; limiar de confiança.
- **Teste de mutação manual (8 mutantes, revertidos):** sobreviveram **7**: remover filtro `min_score`; confiança sempre `"alta"`; citar só 1 fonte; remover overlap; **rotacionar o ranking (top-1 vai para o fim)**; remover sinal do hash; 503→500. Morreu 1 (extrativo devolvendo 1 sentença).
- **Evals:** `evals/cases.json` tem 3 casos e roda contra um CSV sintético de 2 linhas (`test_evals.py:14-20`), não contra o corpus; o caso `out_of_scope` passa por ausência de tokens em comum, não por qualidade do retrieval.
- **Recomendação:** ver R-22 (plano de testes do RAG) + `conftest.py`, fixtures com PDF real gerado (`reportlab`/`pypdf` writer), `httpx.MockTransport` para Ollama, asserções de ranking (`Recall@k`, `MRR`) sobre o corpus real, teste do 503 e do `lifespan`.

#### G-21 · Baixo · `README.md` "Testes e evals" — `pytest -q` falha como documentado
- **Evidência:** sem `conftest.py`/`pyproject.toml`, `pytest -q` → "2 errors during collection" (`ModuleNotFoundError: app`); funciona só com `python -m pytest` ou `PYTHONPATH=.` (o que a CI faz).
- **Recomendação:** `pyproject.toml` com `[tool.pytest.ini_options] pythonpath = ["."]`.

#### G-22 · Baixo · `.github/workflows/ci.yml` — Passo "Check whitespace" é no-op; CI sem lint/type/cobertura/auditoria
- **Evidência:** `git diff --check` sem argumentos num checkout limpo sempre retorna 0 (verificado; a forma que detecta seria `git diff-tree --check --root HEAD` ou `git diff --check origin/main...HEAD`). Só Python 3.12; sem `ruff`, `mypy`, `coverage`, `pip-audit`, build da imagem Docker.
- **Recomendação:** matriz 3.11–3.13 (após decidir versão mínima, §2.4-D6), `ruff check` + `ruff format --check`, `mypy --strict app`, `coverage` com piso, `pip-audit`, `docker build`.

### 2.1.6 Débito técnico e documentação

#### G-23 · Médio · `demo/index.html` — Demo estática diverge do sistema e faz afirmações falsas
- Reimplementa "RAG" em JS com KB hardcoded que **não corresponde ao corpus** (textos diferentes dos PDFs/CSV); afirma que "a versão completa executa … **LangChain**" (não há LangChain); linka `github.com/berger33/Projeto` (repositório antigo). Como o README posiciona o repositório como evidência técnica, a demo enfraquece a credibilidade.
- **Recomendação:** remover, ou marcar explicitamente como mock e apontar para a UI real; decisão em §2.4-D7.

#### G-24 · Médio · `DIAGNOSTICO_WINDOWS.bat:13` — importa `langchain` (não é dependência) → o diagnóstico **sempre falha** com `ModuleNotFoundError` num ambiente correto.

#### G-25 · Baixo · `deploy/OCI_DEPLOY.md`, `deploy/RENDER_DEPLOY.md`, `deploy/oci_compute.sh:5` — clonam `berger33/Projeto` (repositório antigo); `render.yaml` sem `envVars` (roda em modo `local`, contradizendo o README que apresenta `ollama` como caminho real; plano `free` não roda Ollama).

#### G-26 · Baixo · `docker-compose.yml` — sem `environment`/`env_file`, sem serviço `ollama`, sem `healthcheck`; `OLLAMA_BASE_URL=http://127.0.0.1:11434` não funciona de dentro do container.

#### G-27 · Baixo · Ausências de higiene: `pyproject.toml`, `ruff`/`black`/`mypy` config, `pre-commit`, `CHANGELOG`, `SECURITY.md`, `CONTRIBUTING`, `requires-python`; `.gitignore` sem `.coverage`, `htmlcov/`, `.ruff_cache/`, `.mypy_cache/`.

#### G-28 · Baixo · `app/domain.py:32` — `confidence`/`mode` são `str` livres (deveriam ser `Enum`/`Literal`); `SourceRef` não carrega `chunk_id`, `score` nem trecho.

#### G-29 · Baixo · `app/main.py:49,53` — `Depends` em default de argumento (`ruff B008`); `config.py:17` `UP037`. `ruff` com regras E,F,W,B,UP,SIM,N,S: 2 achados reais fora de `S101` em testes.

#### G-30 · Baixo · README overclaims: "A suíte cobre ingestão" (só CSV), "Sem respostas principais hardcoded" (verdadeiro), badge/links ok. `ARQUITETURA.md` lista como "extensões futuras" exatamente os itens que faltam (pgvector, reranking, métricas, tracing) — consistente com esta auditoria.

---

## 2.2 Auditoria específica do RAG

Classificação: **Adequado** / **Precisa Melhorar** / **Ausente**. Cada achado tem ID (R-xx), severidade e localização.

### Checklist

| Item | Classificação | Síntese |
|---|---|---|
| Ingestão | **Precisa Melhorar** | PDF/CSV apenas; versionamento **Ausente**; atualização **Ausente** (exige restart) |
| Chunking | **Precisa Melhorar** (com bugs) | por caractere na prática; 3 defeitos; boilerplate em todo chunk |
| Embeddings | **Precisa Melhorar** | modelo inglês sem prefixos para corpus PT-BR; sem pin; sem persistência; plano de troca **Ausente** |
| Vector store | **Precisa Melhorar** | lista em memória, O(n·d) em Python puro; filtros por metadata **Ausente** |
| Recuperação | **Precisa Melhorar** | k=5 fixo; híbrida **Ausente** (só gate binário); reranking **Ausente** |
| Construção do contexto | **Precisa Melhorar** | sem orçamento de tokens; `num_ctx` não definido; delimitadores injetáveis |
| Rastreabilidade | **Precisa Melhorar** | fontes = top-3 selecionados (não o que o LLM usou); sem citação inline; sem chunk_id/trecho |
| Qualidade/Avaliação | **Precisa Melhorar** → quase Ausente | 3 evals sintéticos; sem Recall/MRR; recusa existe mas detecção frágil |
| Performance e custo | **Precisa Melhorar** | sem medição no modo ollama; cache **Ausente**; reindexa a cada boot |
| Observabilidade | **Ausente** | zero `logging`; sem request id, tempos, scores, prompt |
| Segurança | **Precisa Melhorar** | ACL **Ausente** (corpus público — aceitável se documentado); sem rate limit; injeção só por instrução |
| Testes do pipeline | **Precisa Melhorar** | 8 testes; PDF/Ollama/prompt/503 sem cobertura; 7/8 mutantes sobrevivem |

### Achados detalhados

#### Ingestão

**R-01 · Médio · `app/documents.py:50-61`** — Formatos: só `.pdf` (texto extraível) e `.csv`. Sem `.txt/.md/.html/.docx`, sem OCR, sem subpastas. Versionamento de fontes **Ausente**: nenhum hash/mtime/manifest; `Chunk.id` não muda quando o conteúdo do arquivo muda (mesmo id, texto diferente). Atualização **Ausente**: `lru_cache` segura o serviço até o processo morrer; não há reindex incremental nem endpoint/comando de reload.
→ Recomendação: manifesto de ingestão (`sha256` por arquivo, contagem de chunks, modelo/dimensão do embedding, timestamp), persistido junto com os vetores; reload por comando/CLI; formatos `.md/.txt` (baixo custo, alto valor para documentação interna).

**R-02 · Baixo · `docs/`** — O corpus contém o mesmo FAQ duas vezes (`faq.csv` e `faq.pdf`) com redações levemente diferentes → top-k consumido por quase-duplicatas, fontes "espalhadas" e risco de o LLM ver versões conflitantes. Cabeçalho `"AURORA MODA ONLINE"` e rodapé `"Documento corporativo fictício…"` entram no primeiro/último chunk de cada PDF e poluem embeddings.
→ Recomendação: dedup por hash normalizado / near-dup (Jaccard) na ingestão; remoção de boilerplate configurável; decidir se o CSV ou o PDF é a fonte canônica (§2.4-D8).

#### Chunking

**R-03 · Alto · `app/documents.py:16-44`** — Três defeitos verificados:
1. `Curto.\n\n` + parágrafo de 2.000 chars, `chunk_size=100` → chunks `[6, 2008]`; com 900 → `[6, 5008, 126]`. Parágrafo grande após buffer não vazio (linha 29-32) **nunca cai na janela deslizante** (linhas 33-41 só executam com buffer vazio).
2. `tail(120) + parágrafo(850)` → chunk de **972 > 900** (linha 32 não re-verifica o limite).
3. `_compact` preserva `\n` simples e o `pypdf` nunca emite `\n\n` neste corpus → o split por parágrafo (linha 21) **nunca dispara**; todo PDF é cortado por caractere (linhas 34-40) no meio de palavras: `"5. Como entro em conta"` / `"e acompanhamento do pedido"`; `"tar informações, correções…"` chega literalmente à resposta extrativa.
- Sem esses defeitos ainda faltaria: respeito a fronteiras de sentença/seção (os PDFs têm seções numeradas `1.`–`7.`), medida em tokens (não chars), metadados de seção (`"3. Prazo de entrega"`) no `Chunk`.
→ Recomendação: splitter hierárquico (seção numerada → parágrafo/sentença → janela de tokens) com `chunk_size` ~256–384 tokens e overlap ~15 %; garantir invariantes por teste de propriedade (nenhum chunk > máximo; concatenação cobre todo o texto; nenhum corte intra-palavra).

**R-04 · Médio · `app/documents.py:60`** — Linha do CSV vira `"categoria: X | pergunta: Y | resposta: Z"`. Bom para contexto do LLM, mas: (a) nomes de coluna entram no embedding; (b) no modo local a "sentença" `"categoria: Devolução | pergunta: Qual é o prazo para devolver?"` casa com a pergunta do usuário e é devolvida **como primeira frase da resposta** (eco da pergunta em vez da resposta — observado em 4 de 9 consultas de teste).
→ Recomendação: separar `text` (o que é embedado/mostrado) de `display`/`metadata`; para FAQ, embedar `pergunta + resposta` e apresentar `resposta`.

#### Embeddings

**R-05 · Crítico · `app/embeddings.py:39-55`, `app/config.py:11`** — `nomic-embed-text` (v1.5) **exige** prefixos de tarefa (`search_document:` nos chunks, `search_query:` na pergunta); sem eles o desempenho de recuperação cai (documentação do modelo). O código não envia nenhum prefixo. Além disso, o autor do modelo declara v1.5 como **English-only**; o corpus, as perguntas e o prompt são PT-BR. Alternativas locais no Ollama: `nomic-embed-text-v2-moe` (multilíngue, mesmo fornecedor), `bge-m3`, `qwen3-embedding:0.6b`, `embeddinggemma`. Escolha depende de hardware (§2.4-D1).
→ Recomendação: (i) prefixos configuráveis por provider (`query_prefix`/`document_prefix`); (ii) trocar para modelo multilíngue; (iii) medir com o eval (R-15) antes/depois.

**R-06 · Alto · `app/retrieval.py:26`, `app/main.py:33`** — Sem persistência: **todo boot reembeda o corpus inteiro** no Ollama. Com 13 chunks é irrelevante; com centenas, boot de minutos em CPU e G-08 (timeout 30 s numa única request) derruba o serviço. Plano para troca de modelo **Ausente**: nada registra modelo/dimensão usados; trocar `OLLAMA_EMBED_MODEL` sem reindexar seria detectado só por consulta (G-07).
→ Recomendação: cache de embeddings em disco (`sqlite`/`npy` + manifesto com `modelo`, `dimensão`, `sha256` do texto); invalidar por chave `(modelo, hash do chunk)`; reindex incremental.

**R-07 · Baixo · `app/embeddings.py:18-36`** — `HashEmbeddingProvider`: o bit de sinal `(number>>1)&1` é função determinística do mesmo hash → para um dado índice o sinal é sempre o mesmo (0 conflitos em 200 k tokens) — não reduz colisões como um sign-hash normal. Texto sem tokens produz vetor zero (aceito no índice, nunca recuperável). Adequado como harness; documentar limitação.

#### Vector store

**R-08 · Médio (Alto em escala) · `app/retrieval.py:20-37`** — Busca exaustiva com `zip`/`sum` em Python puro. Medido (d=768): 1.000 chunks → 80 ms/consulta; 5.000 → 421 ms; 20.000 → 1,7 s e ~123 MB como `list[float]`. Sem filtros por metadata (documento, seção, data) — **Ausente**; sem persistência; sem ANN.
→ Recomendação (dimensionamento proposto para a Fase 3): até ~10 k chunks, matriz `numpy` normalizada em memória + persistência (`.npy`/`sqlite`) resolve com <5 ms/consulta; acima disso, backend plugável (`sqlite-vec`, `pgvector` ou `hnswlib`) atrás da mesma interface. Adicionar filtro por `source`/metadata na interface desde já.

#### Recuperação

**R-09 · Alto · `app/rag.py:40-42`** — Ordem errada: `search(k=5)` corta o top-k **antes** do filtro `min_score` + gate lexical. Candidato válido em posição 6 é descartado enquanto posições 1–5 podem ser eliminadas → contexto menor que k mesmo havendo evidência. Verificado com k=2: `faq.csv:r7` (posição 4) falha no gate, `politica_reembolso…:c2` (posição 5) passaria mas nunca é considerado.
→ Recomendação: recuperar `k_candidates` (ex.: 4·k), filtrar, depois cortar em k.

**R-10 · Alto · `app/rag.py:13-17,42`** — Busca híbrida **Ausente**; a componente lexical existe só como gate binário (interseção não vazia). Sem normalização de acentos (`devolucao` ∩ `devolução` = ∅), sem stemming (`reembolsos` ≠ `reembolso`), 18 stopwords (faltam `posso, tenho, fazer, sobre, quando, quanto, onde, dias, pelo, pela, seu, sua…`). Consequência observada em modo local: `"posso pagar com cartao"` → recusa; `"como pago"` → recusa. Em modo Ollama o gate continua sendo o **único** mecanismo que impede citar fontes para perguntas fora de escopo (ver R-11).
→ Recomendação: BM25 (`rank_bm25` ou implementação própria ~60 linhas) sobre texto normalizado (NFKD sem acentos, lowercase, stopwords PT-BR, stemmer RSLP/Snowball-pt opcional) + fusão RRF com o score vetorial; manter o gate como sinal, não como veto.

**R-11 · Alto · `app/config.py:14`, `app/rag.py:42,59`** — Limiares não calibrados para o provider principal. `min_score=0.12` e `alta ≥ 0.45` funcionam para o hash (observado: legítimas 0,26–0,63; fora de escopo até 0,35). Para cosseno de modelos densos a distribuição é deslocada (irrelevantes tipicamente ≥0,3): 0,12 não filtra nada e 0,45 marca quase tudo como "alta". A recusa no modo Ollama fica dependente apenas do gate lexical (R-10) e da boa vontade do LLM (R-13).
→ Recomendação: limiares por provider (ou normalização por z-score/gap entre top-1 e mediana), calibrados com o eval (R-15) e versionados no manifesto.

**R-12 · Médio · `app/rag.py`** — Reranking **Ausente**; sem diversidade (MMR) — quase-duplicatas `faq.csv:r3`/`faq.pdf:p1:c1` ocupam vagas do top-k. `k` fixo, sem adaptação por gap de score.
→ Recomendação: após a fusão híbrida, MMR leve; reranker local opcional (ex.: `bge-reranker-v2-m3` via Ollama/onnx) — decisão de custo (§2.4-D1).

#### Construção do contexto e geração

**R-13 · Crítico · `app/rag.py:51-52`** — Recusa do LLM detectada por `answer.lower().startswith("não encontrei informação suficiente")`. Testado com 7 formulações plausíveis: `"Desculpe, não encontrei…"`, `"Não há informação suficiente…"`, `"A documentação não menciona…"`, `"Infelizmente não encontrei…"`, `"**Não encontrei…**"` (markdown) → **todas recebem fontes e confiança "média"**. Só o prefixo exato e sua versão em caixa alta são detectados. É a violação direta da promessa "recusa antes de citação".
→ Recomendação: saída estruturada do LLM (`format: json` no Ollama com schema `{answer, grounded: bool, used_sources: [n]}`) ou, no mínimo, classificador de recusa robusto (regex de múltiplas formulações + verificação de groundedness por sobreposição resposta↔contexto); tratar recusa como valor de domínio (`RAGAnswer.status`), não string.

**R-14 · Alto · `app/generation.py:16-35,46-53`** — Orçamento de contexto **Ausente**: `build_prompt` concatena k chunks sem limite; `num_ctx` não é enviado (padrão do Ollama 2.048–4.096 tokens conforme versão) → **truncamento silencioso pelo servidor**, cortando o fim do prompt (que é onde estão a PERGUNTA e a instrução RESPOSTA). Simulado: 5 chunks grandes → 200 k chars. Com chunks de 900 chars, k=5 já são ~1,3–1,6 k tokens em PT-BR — perto do limite de 2.048. `num_predict` não definido; `done_reason` não verificado (resposta truncada por `length` é aceita — verificado com mock).
→ Recomendação: contador de tokens aproximado (chars/3,5 para PT ou `tiktoken`-like), orçamento explícito (`num_ctx − prompt_fixo − num_predict`), corte por chunk inteiro com log; enviar `options.num_ctx`/`num_predict`; tratar `done_reason != "stop"`.

**R-15 · Alto · `app/generation.py:38-57`** — Integração com `qwen3` sem controle de *thinking*: o payload não envia `think: false` nem `/no_think`. Conforme versão do Ollama, o modelo (a) gasta tokens raciocinando antes de responder (latência ×2–5 num 0.6B em CPU) e/ou (b) devolve `<think>…</think>` **inline** no `response` — que chegaria intacto ao usuário (verificado com mock: nenhuma limpeza). Usa `/api/generate` (sem papel `system`) em vez de `/api/chat`. Temperatura 0,1 ok. `qwen3:0.6b` é muito pequeno para QA fundamentado em PT-BR com instruções de recusa — decisão de hardware (§2.4-D2).
→ Recomendação: `/api/chat` com `system` + `think: false`; sanitizar `<think>` defensivamente; `keep_alive` configurável; modelo ≥1.7B–4B se o hardware permitir; registrar `eval_count`, `prompt_eval_count`, `total_duration` (o Ollama já devolve) para observabilidade.

**R-16 · Médio · `app/generation.py:20-35`** — Prompt: bom conteúdo (escopo, recusa, não seguir instruções dos docs), mas: rótulos `[FONTE n]` são gerados e **nunca pedidos de volta** (o LLM não é instruído a citar `[n]`, e nada é parseado); delimitadores injetáveis (G-14); sem few-shot de recusa; sem instrução de idioma; prompt não versionado (mudança silenciosa altera evals).
→ Recomendação: pedir citações `[n]` e parseá-las para derivar `sources` (resolve R-17); versionar o template (`prompt_version` no log e na resposta).

#### Rastreabilidade

**R-17 · Alto · `app/rag.py:53-57`** — `sources` = `SourceRef` dos **3 primeiros chunks selecionados**, independentemente do que o LLM usou. Observado: pergunta de pagamento cita `politica_privacidade.pdf` (score 0,31) como 2ª fonte porque passou no gate por compartilhar o token "pagamento". Sem `chunk_id`, `score` ou trecho — impossível auditar "qual frase sustentou a resposta". Sem `request_id` para correlacionar com logs. Página/linha estão corretos quando presentes (**Adequado** nesse aspecto).
→ Recomendação: fontes derivadas das citações do LLM (R-16) com fallback para os selecionados; incluir `chunk_id`, `score`, `excerpt` (≤200 chars) e `request_id` na resposta.

#### Qualidade e avaliação

**R-18 · Alto · `evals/cases.json`, `tests/test_evals.py`** — Sem métricas de recuperação (Recall@k, MRR, nDCG) nem de geração (groundedness, exatidão factual, taxa de recusa correta/indevida). 3 casos, corpus sintético de 2 linhas, asserção por substring. Nenhum caso adversarial (injeção, pergunta parcialmente coberta, sinônimo, erro de digitação, sem acento). Nenhum eval roda contra o corpus real. Tratamento de "não encontrei": existe (R-13), porém frágil.
→ Recomendação: conjunto de ≥40 perguntas sobre o corpus real com `expected_chunk_ids`/`expected_sources`, `must_contain`, `must_not_contain`, categoria (`in_scope`, `out_of_scope`, `partial`, `adversarial`, `typo`, `no_accent`); harness que reporta Recall@5, MRR, precisão de fontes, taxa de recusa correta; roda no modo local na CI e no modo Ollama sob demanda (`pytest -m ollama`).

#### Performance e custo

**R-19 · Médio** — Modo local: 0,9 ms/consulta, boot 23 ms (irrelevante). Modo Ollama: **nunca medido** (nenhum timing no código). Custo monetário zero (local), custo = CPU/GPU e latência do LLM. Cache de consultas **Ausente** (perguntas repetidas — comuns em FAQ — reprocessam embedding + LLM). Um `httpx.Client` novo por chamada (sem keep-alive). Sem limite de concorrência ao Ollama.
→ Recomendação: cache LRU/TTL por pergunta normalizada (+ hash do índice), cliente HTTP reutilizado, semáforo de concorrência, medir e expor `timings` por etapa.

#### Observabilidade

**R-20 · Alto · todo `app/`** — **Ausente.** Zero chamadas a `logging`; nenhuma linha de log em boot (quantos chunks, modelo, dimensão), por consulta (pergunta, top-k com ids/scores, gate, prompt/tamanho, tempos por etapa, `done_reason`, tokens), nem em erro (só o 503 com a mensagem — para o cliente, não para o operador). Sem `request_id`, sem métricas.
→ Recomendação: `logging` estruturado (JSON) com `request_id` (middleware), eventos `index.built`, `query.retrieved`, `query.answered`, `provider.error`; tempos por etapa na resposta (`debug=true` opcional) e em log; opcional OpenTelemetry depois.

#### Segurança

**R-21 · Médio** — Controle de acesso a documentos **Ausente**: todos os chunks são visíveis a qualquer chamador. O corpus atual é de políticas públicas (aceitável) — mas isso não está documentado como premissa, e não há ponto de extensão (metadata `visibility`/`tenant` + filtro em R-08). Demais: G-02 (vazamento de erro), G-13 (sem rate limit), G-14 (injeção), G-12 (root/`.env` na imagem).
→ Recomendação: documentar premissa "corpus público"; reservar campo de metadata para ACL e filtro no retriever; rate limit.

#### Testes do pipeline

**R-22 · Alto** — (ver G-20). Especificamente para RAG faltam: teste de invariantes do chunking; PDF real; providers Ollama via `MockTransport` (contrato, batching, timeout, 404 de modelo, dimensão inconsistente, `<think>`, `done_reason`); ranking sobre o corpus real (Recall@k); recusa em variações; fontes coerentes com a resposta; orçamento de contexto; `lifespan`/readiness; 503 com corpo genérico.

**R-23 · Baixo · `tests/test_rag.py:28`, `tests/test_evals.py:20`** — Fixtures usam `min_score=0.05`, diferente do padrão 0,12 → os testes não exercitam a configuração real.

**R-24 · Baixo · `app/generation.py:69-87`** — Extrativo (harness): sentença-eco da pergunta ranqueada em 1º (R-04); fragmentos de corte intra-palavra (R-03) chegam à resposta; dedup só por igualdade exata (overlap gera quase-duplicatas — observado texto repetido na resposta de "e-mail de privacidade").

**R-25 · Baixo · `app/rag.py:58-59`** — `confidence` tem 3 valores mas só 2 causas (`top_score` e "vazio"); não considera gap top-1/top-2, número de fontes concordantes nem groundedness. Como string livre (G-28).

---

## 2.3 Números de referência coletados

| Métrica | Valor | Contexto |
|---|---|---|
| Chunks no corpus | 13 (6 CSV + 7 PDF) | `load_chunks("docs")` |
| Boot (modo local) | 23 ms | carga + hash embedding |
| Latência consulta (local) | média 0,9 ms · p95 1,0 ms | n=60 |
| Busca exaustiva d=768 | 1 k → 80 ms · 5 k → 421 ms · 20 k → 1,7 s | Python puro, sem numpy |
| Cobertura de linhas | 75 % (branch: 20 parciais) | 8 testes |
| Mutantes sobreviventes | 7 / 8 | ver G-20 |
| Vulnerabilidades (resolvidas) | 0 | `pip-audit` 2026-09-02 |
| Vulnerabilidades (mínimos permitidos) | pypdf 5 · starlette 7 · pytest 1 | idem |
| `ruff` (E,F,W,B,UP,SIM,N,S) | 2 achados fora de testes | B008 ×2, UP037 |

---

## 2.4 Dúvidas e decisões pendentes (para a Fase 3)

Não assumi nenhuma destas; o plano da Fase 3 apresentará opções com trade-off.

- **D1 — Modelo de embedding (Ollama local).** Trocar `nomic-embed-text` (inglês) por multilíngue: `nomic-embed-text-v2-moe` (~475 M params, mesmo fornecedor, prefixos iguais), `bge-m3` (~567 M, 1024 d, forte em PT), `qwen3-embedding:0.6b` (1024 d) ou `embeddinggemma` (300 M, 768 d, leve). Qual hardware roda o Ollama (CPU/GPU, RAM)? Isso define o candidato.
- **D2 — Modelo de geração.** `qwen3:0.6b` é o menor da família; para QA fundamentado em PT-BR com recusa confiável, `qwen3:1.7b`/`qwen3:4b` ou `gemma3:4b` tendem a ser o mínimo prático. Aceita subir o requisito de hardware?
- **D3 — Reranker local.** Vale incluir um reranker (`bge-reranker-v2-m3` ou similar) ao custo de +200–800 ms/consulta em CPU, ou ficar em híbrido + MMR?
- **D4 — Persistência do índice.** Arquivo local (`sqlite`/`.npy` no volume) é suficiente, ou já prever `pgvector`/`sqlite-vec` plugável? (Proposta padrão: local agora, interface plugável.)
- **D5 — Exposição pública.** A API será pública (Render/OCI)? Se sim, rate limit + token simples entram como prioridade Alta; se só uso interno, Média.
- **D6 — Versão mínima de Python.** 3.11 ou 3.12? (CI/Docker = 3.12; código roda em 3.11.)
- **D7 — `demo/index.html`.** Remover, ou reescrever como cliente da API real?
- **D8 — Fonte canônica do FAQ.** Manter `faq.csv` **e** `faq.pdf` (duplicados) ou eleger um? Se ambos, o plano inclui dedup/MMR.
- **D9 — Renomear `docs/` → `corpus/`** (e usar `docs/` para documentação). Quebra links no README e no `main.py:35`; baixo esforço.
- **D10 — Formato de resposta.** Aceita mudar o contrato de `/api/ask` (adicionar `request_id`, `status`, `sources[].chunk_id/score/excerpt`, `timings`) — de forma aditiva, mantendo os campos atuais?

---

## 2.5 Reprodução (comandos usados; não alteram o repositório)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest -q                          # 8 passed
.venv/bin/pytest -q                                                  # falha na coleta (G-21)
.venv/bin/pip install pip-audit ruff coverage
.venv/bin/pip-audit -r requirements.txt                              # 0 vulns (resolvidas)
.venv/bin/ruff check app tests --select E,F,W,B,UP,SIM,C90,N,S --ignore E501
PYTHONPATH=. .venv/bin/coverage run --branch --source=app -m pytest -q && .venv/bin/coverage report -m
# Chunking (R-03):
PYTHONPATH=. .venv/bin/python -c "from app.documents import split_text as s; print([len(c) for c in s('Curto.\n\n'+'x'*2000, chunk_size=100, overlap=20)])"   # [6, 2008]
# Recusa por prefixo (R-13): ver script na seção 2.2 — substituir generator por stub que devolve 'Desculpe, não encontrei…'
```

---

**Fim da Fase 2.** Aguardando sua confirmação para iniciar a Fase 3 (plano de ação priorizado + definição de "RAG completo"). Nenhuma alteração de código foi feita; os arquivos novos no repositório são `auditoria/fase-1-mapeamento.md` (movido de `docs/auditoria/`) e este relatório.
