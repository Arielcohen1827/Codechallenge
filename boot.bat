@echo off
cd /d %~dp0

echo Activando entorno...
call venv\Scripts\activate

echo Ejecutando app...
python run.py eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiQ2hhcm1hbmRlciJ9.LgXm1XLX827-brz8NCN_sZB8RT8ylydUOwgmlAHzihc
