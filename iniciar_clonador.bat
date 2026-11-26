@echo off
title Clonador de Voz Portable AI
chcp 65001 > nul
cls

echo ============================================================
echo   Iniciando Servidor Local de Clonacion de Voz (Portable USB)
echo ============================================================
echo.

REM Establecer el directorio de trabajo relativo a la ubicacion del script (.bat)
cd /d "%~dp0"
set "ROOT_DIR=%~dp0"

REM Configurar variables de entorno para que caches y modelos queden 100%% locales
set "HF_HOME=%ROOT_DIR%backend\models\huggingface"
set "TTS_HOME=%ROOT_DIR%backend\models\tts"
set "TORCH_HOME=%ROOT_DIR%backend\models\torch"
set "XDG_CACHE_HOME=%ROOT_DIR%backend\models\cache"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "TORCHAUDIO_BACKEND=soundfile"

REM Detectar ejecutable de Python portable
set "PYTHON_EXE="
if exist "%ROOT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%venv\Scripts\python.exe"
) else (
    if exist "%ROOT_DIR%venv\python.exe" (
        set "PYTHON_EXE=%ROOT_DIR%venv\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo [!] ERROR: No se encontro el entorno Python portable en "%ROOT_DIR%venv".
    echo [!] Asegurate de que la carpeta 'venv' este presente junto a este archivo.
    echo.
    pause
    exit /b 1
)

REM Agregar Scripts y venv al PATH de la sesion actual
set "PATH=%ROOT_DIR%venv\Scripts;%ROOT_DIR%venv;%PATH%"

echo [✓] Runtime de Python Portable detectado:
echo     %PYTHON_EXE%
echo [✓] Directorio base:
echo     %ROOT_DIR%
echo.

echo [1/2] Verificando modelos locales en backend\models\ ...
"%PYTHON_EXE%" "%ROOT_DIR%backend\download_models.py"
if errorlevel 1 (
    echo.
    echo [!] Advertencia durante la verificacion de modelos, continuando inicio...
)

echo.
echo [2/2] Iniciando servidor FastAPI en puerto 8000 (Local y Red LAN)...
echo [i] La interfaz web se abrira automaticamente en tu navegador predeterminado.
echo [i] Para detener el servidor, presiona Ctrl+C o cierra esta ventana.
echo.
"%PYTHON_EXE%" "%ROOT_DIR%backend\run_server.py"

if errorlevel 1 (
    echo.
    echo [!] El servidor se detuvo con codigo de error %errorlevel%.
)

pause