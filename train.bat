@echo off
rem SlopShield: retrain from your labels (dataset\ai, dataset\real), rebuild the extension.
rem Loop: label while browsing -> run this -> restart the engine + reload the extension -> repeat.
rem A new model is installed only if it wins on the benchmark; then it is also committed and pushed
rem to GitHub (model.onnx via Git LFS). Options are passed through, e.g.:
rem   train.bat --dry-run      report only          train.bat --force     install even if it doesn't win
rem   train.bat --no-push      install, skip git    train.bat --base-gate 0.1
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

"%PY%" scripts\train_from_labels.py %*
set RC=%ERRORLEVEL%
if "%RC%"=="3" (
  echo.
  echo The new model did not beat the current one, so nothing was installed or pushed.
  pause
  exit /b 0
)
if not "%RC%"=="0" (
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
