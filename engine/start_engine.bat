@echo off
rem Start the SlopShield desktop engine (keep this window open while browsing).
rem First run: python -m venv .venv  then  .venv\Scripts\pip install -r engine\requirements.txt
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" engine\slopshield_engine.py %*
) else (
  python engine\slopshield_engine.py %*
)
pause
