@echo off
setlocal
cd /d "%~dp0"
title Aurora Moda - Instalacao e Inicializacao

echo ============================================================
echo  AURORA MODA - INSTALADOR / INICIALIZADOR WINDOWS
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERRO] Python nao foi encontrado no PATH.
        echo Instale Python 3.11 ou superior e tente novamente.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

%PYTHON_CMD% --version
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv
if %errorlevel% neq 0 goto :erro

".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 goto :erro

".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt -r requirements-dev.txt
if %errorlevel% neq 0 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-dev.txt
    if %errorlevel% neq 0 goto :erro
)

".venv\Scripts\python.exe" -m pytest -q
if %errorlevel% neq 0 goto :erro

start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
exit /b 0

:erro
echo.
echo [ERRO] A instalacao nao foi concluida.
pause
exit /b 1
