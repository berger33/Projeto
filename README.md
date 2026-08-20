# Aurora Moda Online - Agente Inteligente para Loja de Roupas

Este repositório recebeu a entrega do Challenge Alura. O projeto completo, com histórico Git local, PDFs, evidências e código, também está no ZIP de entrega.

## Resumo

Agente inteligente para loja de roupas online, capaz de responder dúvidas de clientes com base exclusivamente em documentos corporativos: Política de Privacidade, Política de Reembolso e Devoluções e FAQ.

## Stack

Python, FastAPI, LangChain, Pandas, PyPDF, scikit-learn e Docker.

## Execução

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Acesse `http://localhost:8000`.

## Deploy

O projeto inclui Dockerfile, docker-compose, script OCI Compute e alternativa Render via `render.yaml`.
