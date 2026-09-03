# Deploy na OCI Compute

O projeto está preparado para execução em uma VM Ubuntu da Oracle Cloud Infrastructure usando Docker.
Há dois caminhos:

| Caminho | Comando | O que sobe | Quando usar |
|---|---|---|---|
| **API + Ollama (modo `ollama`)** | `docker compose up -d --build` | API, servidor Ollama e um serviço que baixa `nomic-embed-text-v2-moe` e `qwen3:1.7b` uma única vez | VM com ≥ 8 GB de RAM (em CPU: ≥ 4 vCPUs recomendadas). É o caminho **RAG generativo** |
| **Só a API (modo `local`)** | `./deploy/oci_compute.sh` | imagem isolada, hash embedding + gerador extrativo | VM pequena (1–2 GB); demonstra contrato, retrieval e recusa, **sem LLM** |

## Passos

1. Crie uma instância Ubuntu em **OCI Compute** (shape flex; para o modo `ollama`, 4 OCPU / 16 GB).
2. Libere a porta TCP **80** no Network Security Group/Security List e no firewall da VM
   (`sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT`).
3. Instale Git e Docker (com o plugin `compose`) na instância.
4. Clone o repositório:

```bash
git clone https://github.com/berger33/aurora-document-rag.git
cd aurora-document-rag
```

5. Execute um dos caminhos:

```bash
# (a) API + Ollama — modo generativo
PORT=80 docker compose up -d --build
docker compose logs -f ollama-pull      # aguarde o download dos dois modelos (~2 GB) terminar

# (b) só a API — modo local
chmod +x deploy/oci_compute.sh
./deploy/oci_compute.sh
```

6. Valide:

```text
http://IP_PUBLICO/ready     -> {"ok": true, "checks": {...}}
http://IP_PUBLICO/health
http://IP_PUBLICO/          -> interface (cliente da própria API)
```

7. Para a evidência acadêmica, registre uma captura de tela com a URL/IP público visível e uma resposta do agente
   (o campo `mode` da resposta diz se veio de `ollama` ou de `local`).

## Endurecimento

A API é tratada como não pública por padrão. Numa VM exposta na internet ligue pelo menos:

```bash
# .env na raiz (lido pelo docker compose)
API_TOKEN=um-segredo-com-pelo-menos-16-caracteres
RAG_RATE_LIMIT_PER_MINUTE=30
RAG_DOCS_ENABLED=false
```

e prefira um proxy reverso com TLS (Caddy/nginx) na frente da porta 8000, com `RAG_TRUST_PROXY=true`.

> O código e os scripts estão prontos para OCI, mas uma evidência real de OCI exige uma tenancy/conta Oracle e uma instância efetivamente provisionada.
