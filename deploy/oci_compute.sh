#!/usr/bin/env bash
set -euo pipefail

# Executar em uma VM Ubuntu da OCI Compute após instalar Git e Docker.
git clone "${REPO_URL:-https://github.com/berger33/Projeto.git}" aurora-moda-agente
cd aurora-moda-agente
sudo docker build -t aurora-moda-agente .
sudo docker run -d --name aurora-moda --restart unless-stopped -p 80:8000 aurora-moda-agente
curl -f http://localhost/health
