@echo off
rem SlopShield: retrain from your labels (dataset\ai, dataset\real), rebuild the extension.
rem Loop: label while browsing -> run this -> restart the engine + reload the extension -> repeat.
rem Options are passed through, e.g.:  train.bat --dry-run    or    train.bat --rule head
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

"%PY%" scripts\train_from_labels.py %*
if errorlevel 1 (
  echo.
  echo Training failed - see the messages above. Nothing was changed.
  pause
  exit /b 1
)
echo %* | find "--dry-run" >nul
if not errorlevel 1 (
  pause
  exit /b 0
)

echo.
echo Rebuilding the extension (dist\) with the new model ...
call npm run build --silent
echo.
echo ========================================================================
echo  Next: 1) close the engine window and start engine\start_engine.bat again
echo        2) reload SlopShield in chrome://extensions
echo ========================================================================
pause
