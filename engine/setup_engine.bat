@echo off
rem One-time setup for the SlopShield desktop helper, then starts it.
rem Needs Python 3.10+ (python.org, tick "Add python.exe to PATH"). Safe to run again.
setlocal
cd /d "%~dp0.."

set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul || goto nopython

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment for SlopShield...
  %PY% -m venv .venv || goto fail
)
echo Installing the helper's packages. The first time takes about 2 minutes...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r engine\requirements.txt || goto fail

rem The model is stored with Git LFS: a ZIP download may hold only a small pointer file. Fetch the real one.
set "SIZE=0"
if exist "models\model.onnx" for %%A in ("models\model.onnx") do set "SIZE=%%~zA"
if %SIZE% LSS 1000000 (
  echo Downloading the detector model, about 88 MB...
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://media.githubusercontent.com/media/SteadyStatus21/SlopShield/main/models/model.onnx' -OutFile 'models\model.onnx'" || goto fail
)

echo.
echo Setup done. Starting the helper. Keep this window open while you browse.
echo Next time, just double-click engine\start_engine.bat.
echo.
call "%~dp0start_engine.bat"
goto :eof

:nopython
echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click setup_engine.bat again.
pause
goto :eof

:fail
echo Setup did not finish. Check your internet connection and the messages above, then try again.
pause
