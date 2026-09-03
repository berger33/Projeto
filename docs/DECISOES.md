# Registro de decisões (ADRs curtos)

Decisões tomadas durante a auditoria e o fortalecimento do pipeline (Fases 1–4; relatórios completos em [`auditoria/`](../auditoria/)). Cada entrada: contexto → decisão → consequência.

| # | Decisão | Contexto | Consequência |
|---|---|---|---|
| D1 | Embedding padrão `nomic-embed-text-v2-moe`, prefixos `search_query:`/`search_document:` | `nomic-embed-text` v1.5 é English-only e o código não enviava prefixos (R-05); alvo é CPU sem GPU | prefixos automáticos por família (`app/embeddings.py`); troca de modelo detectada pelo manifesto; alternativas documentadas |
| D2 | Geração padrão `qwen3:1.7b`, `/api/chat`, `think: false`, `num_ctx` 4096, `num_predict` 300 | `qwen3:0.6b` fraco para recusa fundamentada em PT-BR; `<think>` vazava; prompt sem orçamento (R-14/R-15) | orçamento explícito de tokens, saída JSON forçada, `done_reason` verificado |
| D3 | Interface `Reranker` + MMR agora; reranker real só com ganho ≥ 5 p.p. de MRR no eval | cross-encoder em CPU custa 0,3–0,8 s/consulta | `app/rerank.py` com `noop`; MMR por perfil (`local` desligado por medição, `ollama` 0,7 provisório) |
| D4 | Persistência local (`.npy` + `manifest.json`) atrás de `VectorStore` plugável | reembedar a cada boot não escala; pgvector é excesso para o tamanho atual | `app/store.py` + `app/persistence.py`; boot com índice pronto ≈ 20 ms; backend externo é uma classe nova |
| D5 | API tratada como **não pública** | sem informação de exposição externa | token, rate limit e `/docs` desligável existem (P3-03) mas ficam **desligados por padrão** |
| D6 | Python mínimo 3.11; CI em 3.11/3.12/3.13 | `StrEnum`, `datetime.UTC`, sintaxe de tipos | `requires-python >= 3.11`, matriz na CI |
| D7 | `demo/index.html` removida | base de conhecimento falsa, LangChain, link para repo antigo (G-23) | UI real em `/` (`app/static/`) consome a própria API |
| D8 | Manter `faq.csv` **e** `faq.pdf`, com dedup de quase-duplicatas | redações diferentes; o PDF é mais completo | dedup por Jaccard ≥ 0,9 na ingestão (nenhuma duplicata no corpus atual); MMR trata sobreposição no top-k |
| D9 | `docs/` → `corpus/`; `docs/` passa a ser documentação | colisão semântica; a auditoria precisou desviar de `docs/` | `CORPUS_DIR` configurável; este diretório |
| D10 | Contrato de `/api/ask` estendido **só de forma aditiva** | clientes existentes | campos novos: `request_id`, `timings_ms`, `status`, `refusal_reason`, fontes com `chunk_id`/`score`/`section`/`excerpt`/`inferred` |

## Decisões operacionais tomadas pelo eval (Fase 4)

- **Limiares do perfil `local`** calibrados com `evals/calibrate.py` (600 combinações): `min_score` 0,12 mantido por margem contra o máximo do cosseno fora de escopo (0,41); fronteira recusa correta 100 % / indevida 5 %.
- **MMR desligado no perfil `local`**: +2,2 p.p. de selected recall, −6,7 p.p. de precisão de fontes com o hash — o cosseno entre chunks do hash é ruidoso demais para medir redundância.
- **Fontes derivadas do uso real** (`used_sources` → `[n]` → fallback `inferred`): precisão de fontes 0,80 → 0,90 sem tocar no retrieval.

## Dúvidas abertas (exigem ambiente com Ollama)

1. Calibrar o perfil `ollama` (`python -m evals.calibrate --mode ollama`) com `nomic-embed-text-v2-moe`: os valores atuais (min 0,35 / vector-only 0,65 / high 0,7 / MMR 0,7) vêm da escala típica de modelos densos, não de medição.
2. Executar `RAG_EVAL_OLLAMA=1 pytest -m ollama` e registrar `evals/results/*-ollama.json` para fechar os critérios de aceite do plano (§4) no modo principal.
3. Medir latência ponta a ponta com `qwen3:1.7b` no hardware alvo para fixar a meta de p95 (o plano deixou a meta condicionada à baseline).
