@echo off
chcp 65001 >nul
setlocal

set "PYTHON_EXE=C:\Users\WD\anaconda3\envs\dl_study\python.exe"
set "FILE_SIZE="

rem Keep execution rooted at the project directory.
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo [INFO] Check whether the dl_study conda env exists.
    exit /b 1
)

for %%I in (backend\main.py) do set FILE_SIZE=%%~zI
if "%FILE_SIZE%"=="0" (
    echo [INFO] backend\main.py is empty. Server start is skipped.
    exit /b 1
)

"%PYTHON_EXE%" -m uvicorn backend.main:app --reload --reload-dir "%CD%\backend" --reload-dir "%CD%\pipeline" --reload-dir "%CD%\models" --reload-dir "%CD%\data\preprocess" --reload-dir "%CD%\eval"
exit /b %errorlevel%
