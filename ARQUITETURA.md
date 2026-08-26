# Arquitetura — Aurora Document RAG

## Objetivo

Responder perguntas usando exclusivamente a documentação da Aurora, com separação explícita entre ingestão, retrieval, geração e API.

## Componentes

| Componente | Responsabilidade |
|---|---|
| `app/documents.py` | lê PDF/CSV e produz chunks rastreáveis |
| `app/embeddings.py` | converte textos em vetores; provider local ou Ollama |
| `app/retrieval.py` | indexa vetores e calcula similaridade cosseno |
| `app/generation.py` | monta prompt e produz resposta; extrativo local ou LLM Ollama |
| `app/rag.py` | aplica top-k, limiar, relevância, geração e fontes |
| `app/main.py` | expõe contratos HTTP e permite injeção do serviço em testes |

## Fluxo generativo

```mermaid
flowchart LR
    D[PDF/CSV] --> C[Chunks + metadados]
    C --> E[Ollama embeddings]
    E --> V[Índice vetorial]
    Q[Pergunta] --> QE[Embedding da pergunta]
    QE --> V
    V --> R[Top-k contexto]
    R --> P[Prompt fundamentado]
    P --> L[Ollama LLM]
    L --> A[Resposta]
    R --> S[Fontes]
    A --> API[FastAPI]
    S --> API
```

## Por que existe modo local

CI não deve depender de servidor Ollama nem de API paga. Por isso existe um provider vetorial determinístico e uma resposta extrativa. Esse caminho valida ingestão, retrieval, fontes e API; ele **não é apresentado como substituto semântico de um LLM**.

## Guardrails

1. Pergunta vazia é rejeitada.
2. Retrieval usa limiar mínimo e exige evidência de relevância para evitar fontes arbitrárias.
3. O prompt instrui o modelo a tratar documentos apenas como dados e não seguir instruções contidas neles.
4. Sem contexto suficiente, a resposta é de recusa.
5. Fontes são derivadas dos chunks realmente selecionados.
6. Falha do provider gera `503` em vez de sucesso inventado.

## Extensões futuras

- persistir índice em `pgvector`;
- reranking semântico;
- streaming;
- métricas Recall@K/MRR;
- tracing e latência por etapa;
- avaliação de groundedness com conjunto de referência maior.
