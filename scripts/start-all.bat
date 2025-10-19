@echo off
REM ============================================
REM Start All Services - Video Analysis AI
REM ============================================

echo.
echo ============================================
echo   Video Analysis AI - Starting All Services
echo ============================================
echo.
echo This will start:
echo   1. Backend Server (gRPC on port 50051)
echo   2. Frontend Desktop Application (Tauri)
echo.
echo Note: MCP agents spawn on-demand (no separate servers needed)
echo.

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

echo [INFO] Note: For full functionality, start Ollama in another window:
echo [INFO] Command: ollama serve
echo.

REM Check if setup has been run
echo [INFO] Checking if setup is complete...

if not exist "%PROJECT_ROOT%\backend\venv\" (
    echo [ERROR] Backend virtual environment not found!
    echo [INFO] Please run setup-all.bat first.
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%\frontend\node_modules\" (
    echo [ERROR] Frontend dependencies not installed!
    echo [INFO] Please run setup-all.bat first.
    pause
    exit /b 1
)

echo [SUCCESS] Setup appears complete.
echo.

REM Start Backend (gRPC only)
echo ========================================
echo [1/2] Starting Backend Server...
echo ========================================
start "Backend Server (gRPC)" cmd /k "cd /d %PROJECT_ROOT% && scripts\start-backend.bat"
echo [INFO] Backend starting in a new window...
echo [INFO] - gRPC on localhost:50051
echo [INFO] - MCP agents spawn on-demand via stdio
timeout /t 5 >nul

REM Start Frontend
echo.
echo ========================================
echo [2/2] Starting Frontend Application...
echo ========================================
echo [INFO] The desktop application will open shortly...
echo [INFO] First time may take 1-2 minutes to compile...
echo.
cd /d "%PROJECT_ROOT%\frontend"
call npm run tauri:dev

REM This will only execute if the frontend is closed
echo.
echo ========================================
echo   Frontend closed
echo ========================================
echo.
echo To stop the backend, close the Backend Server window.
echo.
pause
