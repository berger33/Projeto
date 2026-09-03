@echo off
cd /d "%~dp0"
echo ===== DIAGNOSTICO AURORA DOCUMENT RAG =====
echo Diretorio: %CD%
echo.
where python
python --version
python -m pip --version
echo.
if exist ".venv\Scripts\python.exe" (
  echo Ambiente virtual encontrado:
  ".venv\Scripts\python.exe" --version
  ".venv\Scripts\python.exe" -m pip check
  rem Importa somente as dependencias reais de runtime (ver pyproject.toml / requirements.txt).
  ".venv\Scripts\python.exe" -c "import fastapi, pypdf, httpx, numpy; print('fastapi', fastapi.__version__); print('pypdf', pypdf.__version__); print('httpx', httpx.__version__); print('numpy', numpy.__version__); print('IMPORTS OK')"
  echo.
  echo Indice persistido:
  ".venv\Scripts\python.exe" -m app.ingest --check
) else (
  echo Ambiente .venv ainda nao existe. Execute INICIAR_WINDOWS.bat.
)
echo.
echo Ollama (opcional, so para RAG_MODE=ollama):
where ollama >nul 2>nul
if %errorlevel%==0 (
  ollama list
) else (
  echo ollama nao encontrado no PATH - o modo local funciona sem ele.
)
echo.
pause
