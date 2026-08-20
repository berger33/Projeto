@echo off
cd /d "%~dp0"
echo ===== DIAGNOSTICO AURORA MODA =====
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
  ".venv\Scripts\python.exe" -c "import pandas, fastapi, pypdf; import langchain; print('pandas', pandas.__version__); print('fastapi', fastapi.__version__); print('langchain', langchain.__version__); print('IMPORTS OK')"
) else (
  echo Ambiente .venv ainda nao existe. Execute INICIAR_WINDOWS.bat.
)
echo.
pause
