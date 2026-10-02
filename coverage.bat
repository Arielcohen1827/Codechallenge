@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

"%PYTHON%" -m coverage erase
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m coverage run -m unittest discover -v
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m coverage report
exit /b %errorlevel%
