# Deploy na OCI Compute

O projeto está preparado para execução em uma VM Ubuntu da Oracle Cloud Infrastructure usando Docker.

## Passos

1. Crie uma instância Ubuntu em **OCI Compute**.
2. Libere a porta TCP **80** no Network Security Group/Security List e no firewall da VM.
3. Instale Git e Docker na instância.
4. Clone o repositório público:

```bash
git clone https://github.com/berger33/Projeto.git aurora-moda-agente
cd aurora-moda-agente
```

5. Execute:

```bash
chmod +x deploy/oci_compute.sh
./deploy/oci_compute.sh
```

6. Valide:

```text
http://IP_PUBLICO/health
http://IP_PUBLICO/
```

7. Para a evidência acadêmica, registre uma captura de tela com a URL/IP público visível e uma resposta do agente.

> O código e os scripts estão prontos para OCI, mas uma evidência real de OCI exige uma tenancy/conta Oracle e uma instância efetivamente provisionada.
