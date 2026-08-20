# Aurora Moda Online — Agente Inteligente Documental

Projeto desenvolvido para o **Challenge Alura Agente**, utilizando como contexto empresarial uma loja de roupas online.

A solução resolve um problema real de atendimento: responder dúvidas recorrentes de clientes sobre **compras, pagamentos, entregas, privacidade, devoluções, reembolsos e suporte**, utilizando exclusivamente a documentação corporativa disponível na base de conhecimento.

## 🚀 Como executar — forma principal e recomendada no Windows

A forma oficial e mais simples de executar o projeto no Windows é pelo arquivo:

```text
INICIAR_WINDOWS.bat
```

### Passo a passo

1. Clone ou baixe este repositório.
2. Abra a pasta do projeto.
3. Dê **duplo clique em `INICIAR_WINDOWS.bat`**.
4. Aguarde a instalação e os testes automáticos.
5. O navegador será aberto em:

```text
http://127.0.0.1:8000
```

O script realiza automaticamente:

- detecção do Python instalado;
- criação do ambiente virtual `.venv`;
- atualização de `pip`, `setuptools` e `wheel`;
- instalação das dependências;
- execução dos testes automatizados;
- inicialização do FastAPI/Uvicorn;
- abertura da aplicação no navegador.

> Recomenda-se Python **3.12, 3.13 ou 3.14**. O projeto foi ajustado e testado para evitar problemas de dependências no Windows/Python 3.14.

Se houver algum problema de ambiente, execute:

```text
DIAGNOSTICO_WINDOWS.bat
```

---

## 🎯 Problema empresarial resolvido

Em um e-commerce de roupas, clientes frequentemente repetem perguntas como:

- Qual é o prazo para devolução?
- O produto pode estar usado?
- Quais formas de pagamento são aceitas?
- Quando começa o prazo de entrega?
- Como os meus dados pessoais são utilizados?
- Como entrar em contato com o suporte?

O agente da **Aurora Moda Online** centraliza essas informações e responde com base apenas nos documentos oficiais do projeto, reduzindo dúvidas repetitivas e evitando respostas fora da política da empresa.

## 📚 Base de conhecimento

A pasta `docs/` contém os documentos utilizados pelo agente:

```text
docs/
├── politica_privacidade.pdf
├── politica_reembolso_devolucoes.pdf
├── faq.pdf
└── faq.csv
```

Os documentos incluem:

- **Política de Privacidade** — coleta, utilização, compartilhamento e proteção de dados;
- **Política de Reembolso e Devoluções** — devolução em até **10 dias corridos após o recebimento**, desde que o produto esteja em perfeitas condições;
- **FAQ** — processo de compra, pagamentos, entrega, rastreamento e suporte.

## 🤖 Agente Inteligente Funcional

O agente implementado em `app/agent.py` realiza um fluxo de recuperação documental (RAG):

1. lê arquivos PDF com **PyPDF**;
2. lê arquivos CSV com **Pandas**;
3. transforma o conteúdo em objetos `Document` do **LangChain**;
4. divide documentos em trechos com `RecursiveCharacterTextSplitter`;
5. indexa os trechos usando **TF-IDF local**;
6. compara a pergunta com a base de conhecimento;
7. recupera os trechos mais relevantes;
8. gera uma resposta fundamentada;
9. apresenta as fontes utilizadas;
10. recusa perguntas quando não existe informação suficiente na documentação.

## 🏗️ Arquitetura

```text
          DOCUMENTAÇÃO CORPORATIVA
        PDF                     CSV
         |                       |
         v                       v
      PyPDF                   Pandas
         \                       /
          \                     /
           v                   v
              LangChain Document
                       |
                       v
          RecursiveCharacterTextSplitter
                       |
                       v
                 Chunks de texto
                       |
                       v
               Índice TF-IDF local
                       |
                       v
                    Retriever
                       |
                       v
              FashionStoreAgent
                 /           \
                /             \
      resposta + fontes     recusa fora da base
                       |
                       v
                    FastAPI
                 /      |      \
                /       |       \
          Interface   /api/ask  /health
```

Uma descrição detalhada também está disponível em `ARQUITETURA.md`.

## 🛠️ Tecnologias e ferramentas

- **Python** — linguagem principal;
- **LangChain** — representação e divisão da documentação;
- **Pandas** — leitura e processamento dos documentos CSV;
- **PyPDF** — extração de conteúdo dos PDFs;
- **TF-IDF** — recuperação e ranqueamento documental local;
- **FastAPI** — aplicação web e API REST;
- **Uvicorn** — servidor ASGI;
- **Pytest** — testes automatizados;
- **GitHub Actions** — integração contínua;
- **Docker / Docker Compose** — empacotamento e execução;
- **OCI Compute** — configuração de deploy em Oracle Cloud;
- **Render** — alternativa de deploy em nuvem.

## 💬 Exemplos de perguntas e respostas

### 1. Devolução

**Pergunta:**

```text
Qual é o prazo para devolver uma roupa?
```

**Resposta:**

```text
Você pode solicitar a devolução em até 10 dias corridos após o recebimento. O produto deve estar em perfeitas condições, sem sinais de uso, lavagem, odores, danos ou alterações, com etiquetas e acessórios originais.
```

### 2. Pagamento

**Pergunta:**

```text
Quais formas de pagamento são aceitas?
```

**Resposta:**

```text
A Aurora Moda Online aceita cartão de crédito e PIX. A confirmação do pedido ocorre após a aprovação do pagamento.
```

### 3. Entrega

**Pergunta:**

```text
Quando começa a contar o prazo de entrega?
```

**Resposta:**

```text
O prazo de entrega começa a contar após a confirmação do pagamento e pode variar conforme CEP, modalidade de frete e transportadora.
```

### 4. Privacidade

**Pergunta:**

```text
Como meus dados pessoais são usados?
```

**Resposta:**

```text
Os dados são utilizados para cadastro, processamento de pedidos, pagamento, entrega, atendimento, segurança e cumprimento de obrigações legais, com medidas técnicas e administrativas de proteção.
```

### 5. Pergunta fora da documentação

**Pergunta:**

```text
Qual é a capital da Austrália?
```

**Resposta:**

```text
Não encontrei essa informação na documentação oficial da Aurora Moda Online. Entre em contato com o suporte para obter orientação adicional.
```

## 🧪 Testes automatizados

O projeto contém testes em `tests/test_agent.py` que validam, entre outros pontos:

- prazo de devolução de 10 dias;
- métodos de pagamento;
- resposta sobre privacidade;
- recusa de perguntas fora da base documental.

Para executar manualmente:

```bash
python -m pytest -q
```

No Windows, o próprio `INICIAR_WINDOWS.bat` executa os testes antes de iniciar a aplicação.

## 🔌 Endpoints

### Interface principal

```text
GET /
```

### Fazer pergunta ao agente

```text
POST /api/ask
```

### Verificar saúde da aplicação

```text
GET /health
```

## 📁 Estrutura do repositório

```text
Projeto/
├── .github/workflows/ci.yml
├── app/
│   ├── __init__.py
│   ├── agent.py
│   └── main.py
├── deploy/
│   ├── OCI_DEPLOY.md
│   ├── RENDER_DEPLOY.md
│   └── oci_compute.sh
├── docs/
│   ├── faq.csv
│   ├── faq.pdf
│   ├── politica_privacidade.pdf
│   └── politica_reembolso_devolucoes.pdf
├── tests/
│   └── test_agent.py
├── .env.example
├── .gitignore
├── ARQUITETURA.md
├── DIAGNOSTICO_WINDOWS.bat
├── Dockerfile
├── docker-compose.yml
├── INICIAR_WINDOWS.bat
├── README.md
├── render.yaml
└── requirements.txt
```

## 🖥️ Execução manual — alternativa

A execução manual é apenas uma alternativa ao `INICIAR_WINDOWS.bat`.

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
docker build -t aurora-moda-agente .
docker run --rm -p 8000:8000 aurora-moda-agente
```

Ou:

```bash
docker compose up --build
```

## ☁️ Deploy

### OCI Compute

O projeto inclui um guia e um script prontos para Oracle Cloud Infrastructure:

```text
deploy/OCI_DEPLOY.md
deploy/oci_compute.sh
```

### Alternativa: Render

Quando não houver conta OCI disponível, o repositório possui:

```text
render.yaml
deploy/RENDER_DEPLOY.md
```

A evidência de um deploy real em nuvem deve ser produzida somente após a publicação efetiva da aplicação; o projeto não utiliza prints ou URLs fictícias.

## 📝 Histórico de desenvolvimento

O repositório utiliza commits separados e descritivos para registrar a evolução do projeto, incluindo etapas como:

- criação inicial e documentação;
- implementação do agente;
- interface FastAPI;
- base de conhecimento;
- Docker e configuração de deploy;
- compatibilidade com Windows/Python 3.14;
- inclusão dos PDFs corporativos;
- correção e estabilização do agente;
- testes automatizados e CI;
- documentação final de entrega.

Isso permite que o histórico do Git demonstre o desenvolvimento progressivo da solução, conforme solicitado no Challenge.

## ✅ Requisitos do Challenge atendidos

- [x] contexto empresarial real: loja de roupas online;
- [x] repositório público no GitHub;
- [x] histórico de commits de desenvolvimento;
- [x] estrutura organizada;
- [x] README com descrição, arquitetura, tecnologias e execução;
- [x] exemplos de perguntas e respostas;
- [x] agente inteligente funcional;
- [x] leitura e processamento de PDF;
- [x] leitura e processamento de CSV com Pandas;
- [x] uso de LangChain no pipeline documental;
- [x] respostas fundamentadas exclusivamente na documentação;
- [x] testes automatizados;
- [x] aplicação web funcional;
- [x] Docker e preparação para deploy em nuvem.

---

**Projeto acadêmico — Challenge Alura Agente | Aurora Moda Online**
