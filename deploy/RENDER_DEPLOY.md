# Alternativa de nuvem: Render

O repositório inclui `render.yaml` para facilitar uma publicação alternativa quando não houver conta OCI disponível.

## Passos

1. Conecte sua conta GitHub ao Render.
2. Crie um novo serviço usando o repositório `berger33/Projeto`.
3. Utilize o Blueprint definido em `render.yaml`.
4. Aguarde o build e a inicialização da aplicação.
5. Valide a URL pública e o endpoint `/health`.
6. Faça uma pergunta na interface e salve uma captura com a URL pública visível.

A configuração de nuvem é complementar; se o avaliador exigir especificamente OCI, siga `OCI_DEPLOY.md`.
