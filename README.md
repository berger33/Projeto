# Aurora Moda Online — Agente Inteligente Documental

Projeto desenvolvido para o **Challenge Alura Agente**, usando como contexto empresarial uma loja de roupas online. A solução responde dúvidas de clientes sobre **compras, pagamentos, entregas, privacidade, devoluções, reembolsos e suporte** com base exclusivamente na documentação corporativa do projeto.

## ▶️ Demo online — experimente sem instalar

**[Abrir o Agente Aurora no navegador](https://htmlpreview.github.io/?https://github.com/berger33/Projeto/blob/main/demo/index.html)**

A demo pública permite conversar com o agente imediatamente, testar perguntas sugeridas, visualizar as fontes recuperadas e confirmar o comportamento de recusa para perguntas fora da documentação.

**[⬇ Baixar o projeto completo](https://github.com/berger33/Projeto/archive/refs/heads/main.zip)** · **[📂 Ver código-fonte](https://github.com/berger33/Projeto)**

> **Transparência:** a demo online executa no navegador uma edição demonstrativa do retrieval documental para permitir avaliação imediata. A aplicação completa deste repositório usa **Python, FastAPI, Pandas, PyPDF e LangChain** e lê os PDFs/CSV reais da pasta `docs/`.

---

## 🚀 Execução completa — forma recomendada no Windows

A forma oficial e mais simples de executar a versão completa é:

```text
INICIAR_WINDOWS.bat
```

1. Clone ou baixe este repositório.
2. Abra a pasta do projeto.
3. Dê duplo clique em `INICIAR_WINDOWS.bat`.
4. Aguarde a instalação e os testes automáticos.
5. A aplicação será aberta em `http://127.0.0.1:8000`.

O script detecta o Python, cria `.venv`, atualiza `pip/setuptools/wheel`, instala dependências, executa os testes e inicia FastAPI/Uvicorn. O projeto é compatível com Python **3.12, 3.13 e 3.14**.

Se houver problema de ambiente, execute `DIAGNOSTICO_WINDOWS.bat`.

---

## 🎯 Problema empresarial resolvido

Em um e-commerce de roupas, clientes repetem dúvidas como:

- Qual é o prazo para devolução?
- O produto pode estar usado?
- Quais formas de pagamento são aceitas?
- Quando começa o prazo de entrega?
- Como os dados pessoais são utilizados?
- Como entrar em contato com o suporte?

O agente centraliza essas informações, reduz perguntas repetitivas e evita respostas fora da política oficial.

## 📚 Base de conhecimento

```text
docs/
├── politica_privacidade.pdf
├── politica_reembolso_devolucoes.pdf
├── faq.pdf
└── faq.csv
```

- **Política de Privacidade** — coleta, utilização, compartilhamento e proteção de dados.
- **Política de Reembolso e Devoluções** — devolução em até **10 dias corridos após o recebimento**, desde que o produto esteja em perfeitas condições.
- **FAQ** — compra, pagamento, entrega, rastreamento e suporte.

## 🤖 Agente inteligente

O pipeline em `app/agent.py`:

1. lê PDF com **PyPDF**;
2. lê CSV com **Pandas**;
3. transforma o conteúdo em `Document` do **LangChain**;
4. divide documentos com `RecursiveCharacterTextSplitter`;
5. indexa os trechos usando **TF-IDF local**;
6. compara a pergunta com a base;
7. recupera os trechos mais relevantes;
8. gera resposta fundamentada;
9. mostra as fontes utilizadas;
10. recusa a pergunta quando não há informação suficiente.

## 🏗️ Arquitetura

```text
PDF ──PyPDF──┐
             ├─> LangChain Documents -> Chunks -> TF-IDF -> Retriever
CSV ─Pandas──┘                                      |
                                                     v
                                            FashionStoreAgent
                                               /          \
                                      resposta+fontes   recusa
                                               |
                                               v
                                             FastAPI
```

Detalhes adicionais: [`ARQUITETURA.md`](ARQUITETURA.md).

## 🛠️ Tecnologias

`Python` · `FastAPI` · `LangChain` · `Pandas` · `PyPDF` · `TF-IDF` · `Uvicorn` · `Pytest` · `GitHub Actions` · `Docker` · `Docker Compose`

## 💬 Exemplos

### Devolução

**Pergunta:** `Qual é o prazo para devolver uma roupa?`

**Resposta esperada:** você pode solicitar a devolução em até **10 dias corridos após o recebimento**, e o produto deve estar em perfeitas condições, sem sinais de uso, lavagem, odores, danos ou alterações, com etiquetas e acessórios originais.

### Pagamento

**Pergunta:** `Quais formas de pagamento são aceitas?`

**Resposta esperada:** cartão de crédito e PIX.

### Entrega

**Pergunta:** `Quando começa a contar o prazo de entrega?`

**Resposta esperada:** após a confirmação do pagamento, variando conforme CEP, modalidade de frete e transportadora.

### Privacidade

**Pergunta:** `Como meus dados pessoais são usados?`

**Resposta esperada:** para cadastro, pedido, pagamento, entrega, atendimento, segurança e obrigações legais, com medidas de proteção.

### Fora da documentação

**Pergunta:** `Qual é a capital da Austrália?`

**Resposta:** `Não encontrei essa informação na documentação oficial da Aurora Moda Online.`

## 🧪 Testes

```bash
python -m pytest -q
```

Os testes cobrem devolução, pagamentos, privacidade, perguntas fora da base e health check. No Windows, `INICIAR_WINDOWS.bat` executa os testes antes de iniciar a aplicação.

## 🔌 Endpoints

| Endpoint | Função |
| --- | --- |
| `GET /` | interface principal |
| `POST /api/ask` | pergunta ao agente |
| `GET /health` | saúde da aplicação |

## 📁 Estrutura

```text
Projeto/
├── .github/workflows/ci.yml
├── app/
│   ├── agent.py
│   └── main.py
├── demo/
│   └── index.html
├── deploy/
├── docs/
│   ├── faq.csv
│   ├── faq.pdf
│   ├── politica_privacidade.pdf
│   └── politica_reembolso_devolucoes.pdf
├── tests/
├── ARQUITETURA.md
├── DIAGNOSTICO_WINDOWS.bat
├── Dockerfile
├── docker-compose.yml
├── INICIAR_WINDOWS.bat
├── render.yaml
└── requirements.txt
```

## 🖥️ Execução manual

### Windows

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🐳 Docker

```bash
docker compose up --build
```

## ☁️ Deploy

O repositório mantém guias para OCI Compute e Render em `deploy/`. A demo pública existe para avaliação imediata do produto; um deploy real de backend deve ser documentado somente quando efetivamente publicado.

## 📝 Histórico de desenvolvimento

O histórico Git registra etapas separadas de implementação, documentação, compatibilidade Windows/Python 3.14, base documental, Docker, testes/CI, correções e criação da demo pública. Isso preserva a evolução do projeto em vez de apresentar um único upload final.

## ✅ Challenge Alura — requisitos atendidos

- [x] contexto empresarial real;
- [x] repositório público e organizado;
- [x] histórico de commits;
- [x] README com descrição, arquitetura, tecnologias e execução;
- [x] exemplos de perguntas e respostas;
- [x] agente funcional;
- [x] leitura de PDF e CSV;
- [x] Pandas e LangChain no pipeline;
- [x] respostas limitadas à documentação;
- [x] testes automatizados e CI;
- [x] aplicação web e Docker;
- [x] demo pública navegável sem instalação.

---

**Projeto acadêmico — Challenge Alura Agente | Aurora Moda Online**