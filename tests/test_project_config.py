"""Testes de configuração do projeto (P0-01).

Garantem que `pyproject.toml`, `uv.lock` e os `requirements*.txt` exportados permanecem coerentes,
que dependências de desenvolvimento não vazam para o runtime e que os pisos de segurança
definidos na auditoria (Fase 2, G-11) não regridem silenciosamente.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def _pinned_versions(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", line.strip())
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _lock_versions(lock: dict) -> dict[str, str]:
    return {pkg["name"].lower(): pkg["version"] for pkg in lock["package"] if "version" in pkg}


def test_python_floor_is_declared(pyproject: dict) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.11"


def test_pytest_runs_from_repo_root_without_pythonpath(pyproject: dict) -> None:
    # Fase 2, G-21: `pytest -q` falhava na coleta porque `app` não estava no sys.path.
    assert "." in pyproject["tool"]["pytest"]["ini_options"]["pythonpath"]


def test_dev_tools_are_not_runtime_dependencies(pyproject: dict) -> None:
    runtime = _requirement_names("\n".join(pyproject["project"]["dependencies"]))
    dev = _requirement_names("\n".join(pyproject["dependency-groups"]["dev"]))
    assert {"pytest", "ruff", "mypy", "coverage", "pip-audit"} <= dev
    assert not ({"pytest", "ruff", "mypy", "coverage", "pip-audit"} & runtime)


def test_exported_runtime_requirements_exclude_dev_tools() -> None:
    names = _requirement_names((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert {"fastapi", "uvicorn", "pypdf", "httpx"} <= names
    assert not ({"pytest", "ruff", "mypy", "coverage", "pip-audit"} & names)


def test_exported_requirements_match_lockfile(lock: dict) -> None:
    locked = _lock_versions(lock)
    for filename in ("requirements.txt", "requirements-dev.txt"):
        pins = _pinned_versions((ROOT / filename).read_text(encoding="utf-8"))
        assert pins, f"{filename} não contém pins"
        for name, version in pins.items():
            assert locked.get(name) == version, f"{filename}: {name}=={version} diverge do uv.lock ({locked.get(name)})"


@pytest.mark.parametrize(
    ("package", "minimum"),
    [
        ("pypdf", (6, 16, 1)),  # GHSA-jm82-fx9c-mx94, CVE-2026-84309/84310/84311, CVE-2026-82398
        ("starlette", (1, 3, 1)),  # PYSEC-2026-161/248/249/1941/1942/2280/2281
        ("fastapi", (0, 135, 0)),  # primeira série que não trava starlette < 1.0
    ],
)
def test_security_floors_are_not_lowered(lock: dict, package: str, minimum: tuple[int, ...]) -> None:
    version = _lock_versions(lock)[package]
    parsed = tuple(int(part) for part in re.findall(r"\d+", version)[: len(minimum)])
    assert parsed >= minimum, f"{package}=={version} está abaixo do piso de segurança {minimum}"


def test_lockfile_covers_all_declared_dependencies(pyproject: dict, lock: dict) -> None:
    locked = set(_lock_versions(lock))
    declared = _requirement_names("\n".join(pyproject["project"]["dependencies"]))
    declared |= _requirement_names("\n".join(pyproject["dependency-groups"]["dev"]))
    missing = {name for name in declared if name not in locked}
    assert not missing, f"dependências declaradas sem entrada no uv.lock: {missing}"


# ---------------------------------------------------------------------------
# P3-01: CI completa (G-22)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ci_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ci_runs_full_matrix_and_all_quality_gates(ci_workflow: str) -> None:
    assert 'python-version: ["3.11", "3.12", "3.13"]' in ci_workflow
    for step in (
        "ruff check app evals tests",
        "ruff format --check app evals tests",
        "mypy app evals",
        "coverage run -m pytest -q",
        "coverage report",
        "pip-audit -r requirements.txt",
        "uv lock --check",
        "docker build -t aurora-document-rag:ci .",
        "python -m evals.run --mode local",
        "python -m app.ingest --index-dir /tmp/ci-index --check",
    ):
        assert step in ci_workflow, step


def test_ci_whitespace_check_compares_against_pr_base(ci_workflow: str) -> None:
    """G-22: o passo antigo de whitespace era um no-op (comparava HEAD com HEAD)."""
    assert 'git diff --check "origin/${{ github.base_ref }}...HEAD"' in ci_workflow
    assert "fetch-depth: 0" in ci_workflow


def test_ci_docker_job_boots_container_and_probes_ready(ci_workflow: str) -> None:
    assert "docker run -d --name aurora" in ci_workflow
    assert "curl -fsS http://127.0.0.1:8000/ready" in ci_workflow
    assert '"status":"answered"' in ci_workflow


def test_mypy_is_strict_over_app_and_evals(pyproject: dict) -> None:
    mypy = pyproject["tool"]["mypy"]
    assert mypy["strict"] is True and set(mypy["files"]) == {"app", "evals"}


def test_coverage_floor_is_at_least_85(pyproject: dict) -> None:
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] >= 85


# ---------------------------------------------------------------------------
# P3-02: Docker/compose (G-12, G-26)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_runs_as_non_root_with_healthcheck_and_prebuilt_index(dockerfile: str) -> None:
    assert "USER aurora" in dockerfile and "useradd --system --uid 10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/health" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert (
        "COPY --chown=aurora:aurora app/ ./app/" in dockerfile
        and "COPY --chown=aurora:aurora corpus/ ./corpus/" in dockerfile
    )
    assert "COPY . ." not in dockerfile
    assert "RUN python -m app.ingest --index-dir /data/index" in dockerfile
    assert 'VOLUME ["/data/index"]' in dockerfile and "RAG_INDEX_DIR=/data/index" in dockerfile
    # O pip install acontece antes de copiar o código: cache de camadas preservado entre mudanças no app.
    assert dockerfile.index("pip install") < dockerfile.index("COPY --chown=aurora:aurora app/")


def test_dockerignore_whitelists_only_runtime_inputs() -> None:
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "*"
    assert {"!app/", "!corpus/", "!requirements.txt"} <= set(rules)
    assert not any(rule.startswith("!") and rule not in {"!app/", "!corpus/", "!requirements.txt"} for rule in rules)


def test_compose_runs_ollama_mode_with_model_pull_and_volumes() -> None:
    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"app", "ollama", "ollama-pull"}
    app = services["app"]
    assert app["environment"]["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert app["environment"]["RAG_INDEX_DIR"] == "/data/index" and "rag-index:/data/index" in app["volumes"]
    assert app["depends_on"]["ollama-pull"]["condition"] == "service_completed_successfully"
    assert "/ready" in " ".join(app["healthcheck"]["test"])
    assert any(entry.get("path") == ".env" and entry.get("required") is False for entry in app["env_file"])
    assert "ollama-models:/root/.ollama" in services["ollama"]["volumes"]
    pull = " ".join(services["ollama-pull"]["command"])
    assert "ollama pull" in pull and "nomic-embed-text-v2-moe" in pull and "qwen3:1.7b" in pull
    assert set(compose["volumes"]) == {"ollama-models", "rag-index"}


# ---------------------------------------------------------------------------
# P3-05: corpus/ vs docs/ (D9)
# ---------------------------------------------------------------------------


def test_corpus_lives_in_corpus_dir_and_docs_holds_documentation() -> None:
    assert (ROOT / "corpus" / "faq.csv").exists() and not (ROOT / "docs" / "faq.csv").exists()
    for name in ("ARQUITETURA.md", "OPERACAO.md", "DECISOES.md"):
        assert (ROOT / "docs" / name).exists(), name
    assert not (ROOT / "ARQUITETURA.md").exists()


def test_corpus_dir_is_configurable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings
    from app.main import build_service

    assert Settings().corpus_dir == "corpus"
    corpus = tmp_path / "meu_corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("O prazo de devolução é de 10 dias corridos após o recebimento.", encoding="utf-8")
    monkeypatch.setenv("RAG_MODE", "local")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    service = build_service()
    assert service.docs_dir == corpus and service.chunk_count == 1
    assert Settings.from_env({"CORPUS_DIR": "corpus"}).public_dict()["corpus_dir"] == "corpus"


# ---------------------------------------------------------------------------
# P3-06: higiene de scripts, deploy e documentação (G-24, G-25, G-30)
# ---------------------------------------------------------------------------


def _runtime_dependency_names(pyproject: dict) -> set[str]:
    return {
        dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
        for dep in pyproject["project"]["dependencies"]
    }


def test_windows_diagnostic_imports_only_real_runtime_dependencies(pyproject: dict) -> None:
    """G-24: o diagnóstico importava langchain/pandas (não são dependências) e falhava sempre."""
    script = (ROOT / "DIAGNOSTICO_WINDOWS.bat").read_text(encoding="utf-8")
    imports = re.findall(r"import ([a-zA-Z_, ]+);", script)
    assert imports, "o diagnóstico deve testar os imports de runtime"
    imported = {name.strip() for group in imports for name in group.split(",")}
    assert "langchain" not in imported and "pandas" not in imported
    assert imported <= _runtime_dependency_names(pyproject) | {"numpy"}
    assert "-m app.ingest --check" in script


def test_windows_scripts_do_not_reference_removed_tooling() -> None:
    for name in ("DIAGNOSTICO_WINDOWS.bat", "INICIAR_WINDOWS.bat"):
        script = (ROOT / name).read_text(encoding="utf-8")
        assert "langchain" not in script and "pandas" not in script, name


def test_deploy_docs_point_to_current_repository_and_ready_probe() -> None:
    """G-25: deploy/* clonava berger33/Projeto (repositório antigo) e sondava só /health."""
    for path in (
        ROOT / "deploy" / "OCI_DEPLOY.md",
        ROOT / "deploy" / "RENDER_DEPLOY.md",
        ROOT / "deploy" / "oci_compute.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "berger33/Projeto" not in text, path.name
        assert "berger33/aurora-document-rag" in text, path.name
        assert "/ready" in text, path.name


def test_oci_script_is_executable_and_uses_ready_probe() -> None:
    import shutil
    import subprocess

    script = ROOT / "deploy" / "oci_compute.sh"
    assert script.stat().st_mode & 0o111, "deploy/oci_compute.sh precisa ser executável"
    bash = shutil.which("bash")
    if bash:  # sintaxe válida (o script não é executado)
        assert subprocess.run([bash, "-n", str(script)], capture_output=True, check=False).returncode == 0  # noqa: S603
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    assert '-p "${HOST_PORT}:8000"' in text
    assert 'curl -fsS "http://localhost:${HOST_PORT}/ready"' in text


def test_render_blueprint_runs_local_mode_with_explicit_env_and_ready_check() -> None:
    import yaml

    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    (service,) = blueprint["services"]
    assert service["name"] == "aurora-document-rag" and service["runtime"] == "docker"
    assert service["healthCheckPath"] == "/ready"
    env = {item["key"]: item.get("value") for item in service["envVars"]}
    assert env["RAG_MODE"] == "local" and env["RAG_INDEX_DIR"] == "/data/index"
    # O plano free não roda Ollama: o blueprint e o guia precisam dizer isso explicitamente.
    assert service["plan"] == "free"
    guide = (ROOT / "deploy" / "RENDER_DEPLOY.md").read_text(encoding="utf-8")
    assert "modo `local`" in guide and "não" in guide.lower() and "Ollama" in guide


def test_readme_has_limitations_and_no_overclaims() -> None:
    """G-30: README dizia cobrir ingestão de PDF (só CSV) e listava Pandas na stack."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Limitações" in readme
    limitations = readme.split("## Limitações", 1)[1].split("\n## ", 1)[0]
    for topic in (
        "modo `local` não é semântico",
        "perfil `ollama` é provisório",
        "Sem OCR",
        "Rate limit por processo",
        "R-07",
    ):
        assert topic in limitations, topic
    assert "Pandas" not in readme and "pandas" not in readme
    assert "tests/test_rag.py" not in readme  # a tabela de evidências aponta para a suíte inteira
    for link in (
        "SECURITY.md",
        "CHANGELOG.md",
        "deploy/OCI_DEPLOY.md",
        "deploy/RENDER_DEPLOY.md",
        "DIAGNOSTICO_WINDOWS.bat",
    ):
        assert link in readme, link


def test_security_policy_and_changelog_exist_and_are_consistent(pyproject: dict) -> None:
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    for topic in ("Reportar", "API_TOKEN", "pip-audit", "não-root", "Prompt injection"):
        assert topic in security, topic
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.startswith("# Changelog")
    assert "## [2.1.0]" in changelog and "## [2.0.0]" in changelog
    # Cada item do plano implementado aparece no changelog.
    for item in (
        "P0-01",
        "P0-02",
        "P0-03",
        "P0-04",
        "P1-01",
        "P1-02",
        "P1-03",
        "P1-04",
        "P1-05",
        "P1-06",
        "P2-01",
        "P2-02",
        "P2-03",
        "P2-04",
        "P2-05",
        "P2-06",
        "P3-01",
        "P3-02",
        "P3-03",
        "P3-04",
        "P3-05",
        "P3-06",
    ):
        assert item in changelog, item
    # A versão anunciada pela API e pelo pacote é a documentada.
    main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert 'version="2.1.0"' in main_src and pyproject["project"]["version"] == "2.1.0"
