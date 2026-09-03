# Segurança

## Reportar uma vulnerabilidade

Abra um [security advisory privado](https://github.com/berger33/aurora-document-rag/security/advisories/new)
no GitHub (preferencial) ou uma issue **sem detalhes exploráveis** pedindo contato. Inclua versão/commit,
passos para reproduzir e impacto esperado. Projeto acadêmico mantido por uma pessoa: a resposta inicial
pode levar alguns dias; correções saem na branch `main` e são registradas no `CHANGELOG.md`.

Só a versão em `main` recebe correções.

## Modelo de ameaça (o que a aplicação assume)

- A API é tratada como **não pública** por padrão (decisão D5): autenticação, rate limit e ocultação
  do OpenAPI existem, mas ficam **desligados** até serem configurados. Quem expõe a porta na internet
  deve ligar `API_TOKEN` (≥ 16 caracteres), `RAG_RATE_LIMIT_PER_MINUTE` e, atrás de proxy,
  `RAG_TRUST_PROXY=true` — ver `docs/OPERACAO.md` e `deploy/`.
- O **corpus é público** (políticas de uma loja fictícia). Não há ACL por documento; o filtro por
  metadado do `VectorStore` é o ponto de extensão, não uma implementação.
- O **Ollama é um serviço interno**: `OLLAMA_BASE_URL` deve apontar para a rede local/compose. A
  aplicação nunca repassa erros do Ollama ao cliente (só `error_code` estável + `request_id`).

## Controles implementados

| Área | Controle |
|---|---|
| Entrada | pergunta limitada a 2–2000 caracteres, caracteres de controle rejeitados (422); JSON validado por Pydantic |
| Prompt injection | trechos do corpus e a pergunta entram em blocos delimitados com `<` escapado; o gerador responde em JSON e a resposta passa por um juiz de recusa e verificação de sustentação antes de receber fontes; casos `adversarial` no eval (100 % de resistência exigida) |
| Segredos | nada de credencial no repositório; `.env` ignorado pelo Git e pelo `.dockerignore`; `settings.loaded` loga a configuração **sem** token e com a URL do Ollama sem credenciais; `API_TOKEN` comparado em tempo constante |
| Autenticação/abuso (opcional) | `Authorization: Bearer` em `/api/*`, token bucket por IP com `Retry-After`, `/docs` desligável |
| Logs | texto de pergunta/resposta só em `DEBUG` (podem conter dados pessoais); eventos estruturados com `request_id` |
| Dependências | pisos com advisories conhecidos travados em `pyproject.toml` (`pypdf ≥ 6.16.1`, `starlette ≥ 1.3.1`); `uv.lock` + `requirements*.txt` exportados; `pip-audit -r requirements.txt` na CI; lockfile verificado a cada PR |
| Imagem | `python:3.12-slim`, usuário não-root (`aurora`, uid 10001), só `app/` + `corpus/` + `requirements.txt` copiados, `HEALTHCHECK` |
| Falhas | erro de provider vira `503` com `error_code`; stack trace fica no log, nunca na resposta |

## Fora do escopo / limitações conhecidas

- Rate limit é **por processo** (sem estado compartilhado entre workers/réplicas).
- Não há TLS na aplicação: termine TLS num proxy reverso.
- Não há proteção CSRF porque a API não usa cookies/sessão.
- Um PDF malicioso pode consumir CPU/memória na ingestão (`pypdf`); a ingestão roda no boot ou via
  `python -m app.ingest`, nunca a partir de upload de usuário.
