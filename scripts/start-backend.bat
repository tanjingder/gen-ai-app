@echo off
REM ============================================
REM Start Backend - Video Analysis AI
REM Runs gRPC server only (port 50051)
REM ============================================

echo.
echo ========================================
echo   Video Analysis AI - Starting Backend
echo ========================================
echo.
echo This starts:
echo   - gRPC Server on localhost:50051 (Tauri frontend connects here)
echo.

cd /d "%~dp0..\backend"

REM Check if venv exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo [INFO] Please run setup-backend.bat first.
    pause
    exit /b 1
)

echo [INFO] Recommended: Start Ollama in another window
echo [INFO] Command: ollama serve
echo.
echo [INFO] Starting gRPC backend server...
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Activate virtual environment and start backend
call venv\Scripts\activate.bat
python main.py

pause
