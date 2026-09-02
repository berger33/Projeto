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
