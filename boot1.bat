@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Error: falta venv\Scripts\python.exe. Crea el entorno virtual antes de iniciar.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" launch_bot.py
set "launch_status=%errorlevel%"
if not "%launch_status%"=="0" pause
exit /b %launch_status%
