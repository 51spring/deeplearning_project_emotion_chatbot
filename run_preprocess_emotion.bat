@echo off
chcp 65001 >nul
setlocal

set "PYTHON_EXE=C:\Users\WD\anaconda3\envs\dl_study\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo [INFO] Check whether the dl_study conda env exists.
    exit /b 1
)

"%PYTHON_EXE%" data\preprocess\preprocess_emotion.py
exit /b %errorlevel%
