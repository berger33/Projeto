# Arquitetura da Solução

## Visão geral

A Aurora Moda Online utiliza uma arquitetura RAG documental local para responder dúvidas de clientes com base exclusiva nos documentos corporativos do projeto.

```text
PDFs / CSV
   |
   v
PyPDF + Pandas
   |
   v
LangChain Document
   |
   v
RecursiveCharacterTextSplitter
   |
   v
Chunks documentais
   |
   v
Índice TF-IDF local
   |
   v
Retriever por similaridade + sobreposição de termos
   |
   v
Agente documental
   |
   +--> resposta fundamentada + fontes
   |
   +--> recusa quando a informação não está na base
   |
   v
FastAPI
   |
   +--> Interface Web
   +--> POST /api/ask
   +--> GET /health
```

## Componentes

1. **PyPDF** extrai texto dos documentos PDF.
2. **Pandas** lê e transforma o FAQ em CSV.
3. **LangChain** representa o conteúdo como `Document` e realiza a divisão em trechos com `RecursiveCharacterTextSplitter`.
4. **TF-IDF em Python puro** cria a representação local dos chunks sem depender de uma API externa.
5. **Retriever** calcula relevância entre a pergunta e os trechos da documentação.
6. **FashionStoreAgent** responde somente quando existe evidência documental suficiente.
7. **FastAPI** disponibiliza interface web, API e health check.
8. **Docker** empacota a aplicação para execução local ou em nuvem.

## Base de conhecimento

A pasta `docs/` contém:

- `politica_privacidade.pdf`
- `politica_reembolso_devolucoes.pdf`
- `faq.pdf`
- `faq.csv`

## Decisão de projeto

A aplicação foi construída para ser reproduzível durante a avaliação acadêmica sem exigir chave de API paga. O pipeline continua demonstrando ingestão, chunking, indexação, recuperação e geração de resposta fundamentada, com LangChain participando diretamente do processamento documental.

## Deploy

O projeto contém `Dockerfile`, `docker-compose.yml`, `render.yaml` e scripts/instruções em `deploy/` para OCI Compute e Render.
