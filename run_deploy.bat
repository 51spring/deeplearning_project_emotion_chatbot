@echo off
setlocal

rem Secure class demo single-server deployment.
rem External binding requires explicit environment credentials without fixed fallbacks.
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\WD\anaconda3\envs\dl_study\python.exe"
set "PYTHONPATH=%CD%"

if not defined EMOTION_CHATBOT_HOST set "EMOTION_CHATBOT_HOST=0.0.0.0"
if not defined EMOTION_CHATBOT_PORT set "EMOTION_CHATBOT_PORT=8000"
set "EMOTION_CHATBOT_ENV=production"

:REQUIRE_SECURE_ENV
if not defined EMOTION_CHATBOT_AUTH_SECRET (
    echo [ERROR] EMOTION_CHATBOT_AUTH_SECRET is required for deployment.
    exit /b 1
)
if not defined EMOTION_CHATBOT_DEVELOPER_PASSWORD (
    echo [ERROR] EMOTION_CHATBOT_DEVELOPER_PASSWORD is required for deployment.
    exit /b 1
)
if not defined EMOTION_CHATBOT_ROOT_PASSWORD (
    echo [ERROR] EMOTION_CHATBOT_ROOT_PASSWORD is required for deployment.
    exit /b 1
)
:ENV_READY
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo [INFO] Check the dl_study conda env and official Python path.
    exit /b 1
)

echo [INFO] Validating production authentication settings...
"%PYTHON_EXE%" -c "import backend.auth_utils; import backend.db.crud"
if errorlevel 1 (
    echo [ERROR] production authentication settings are invalid.
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm command was not found. Check Node.js/npm and PATH.
    exit /b 1
)

echo [INFO] Building React production bundle...
call npm --prefix frontend run build
if errorlevel 1 (
    echo [ERROR] frontend build failed.
    exit /b 1
)

if "%EMOTION_CHATBOT_DEPLOY_DRY_RUN%"=="1" (
    echo [INFO] dry-run mode: build/env check passed; server start skipped.
    exit /b 0
)

echo [INFO] Starting deploy server: http://%EMOTION_CHATBOT_HOST%:%EMOTION_CHATBOT_PORT%
echo [INFO] Same-network URL: http://^<PC-IP^>:%EMOTION_CHATBOT_PORT%
echo [INFO] For external access, tunnel localhost:%EMOTION_CHATBOT_PORT% with ngrok/cloudflared.

"%PYTHON_EXE%" -m uvicorn backend.main:app --host "%EMOTION_CHATBOT_HOST%" --port "%EMOTION_CHATBOT_PORT%"
exit /b %errorlevel%
