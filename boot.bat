@echo off
cd /d %~dp0

echo Activando entorno...
call venv\Scripts\activate

echo Ejecutando app...
python run.py eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYXJpZWxjb2hlbiJ9.7xIgqanz4410M1ETwxgDOWB8Gbd-kD0PEVxSfg-Ef4Q
