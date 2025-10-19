@echo off
REM ============================================
REM Complete Setup - Video Analysis AI
REM ============================================

echo.
echo ============================================
echo   Video Analysis AI - Complete Setup
echo ============================================
echo.
echo This will set up all components:
echo - Backend (Python + gRPC + Shared venv)
echo - MCP Servers (3 Agents using shared venv)
echo - Frontend (React + Tauri)
echo.
echo This may take several minutes...
echo.
pause

cd /d "%~dp0.."

REM Setup Backend
echo.
echo ========================================
echo   Step 1/3: Setting up Backend
echo ========================================
call scripts\setup-backend.bat
if errorlevel 1 (
    echo [ERROR] Backend setup failed!
    pause
    exit /b 1
)

cd /d "%~dp0.."

REM Setup MCP Servers
echo.
echo ========================================
echo   Step 2/3: Setting up MCP Servers
echo ========================================
call scripts\setup-mcp-servers.bat
if errorlevel 1 (
    echo [ERROR] MCP Servers setup failed!
    pause
    exit /b 1
)

cd /d "%~dp0.."

REM Setup Frontend
echo.
echo ========================================
echo   Step 3/3: Setting up Frontend
echo ========================================
call scripts\setup-frontend.bat
if errorlevel 1 (
    echo [ERROR] Frontend setup failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Complete Setup Finished!
echo ============================================
echo.
echo All components are now set up!
echo.
echo IMPORTANT: Before running the application:
echo 1. Install Ollama from https://ollama.ai
echo 2. Run: ollama serve
echo 3. Run: ollama pull llama3.2
echo 4. Run: ollama pull llava
echo.
echo Then use start-all.bat to start everything!
echo.
pause
