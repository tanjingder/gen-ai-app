@echo off
REM ============================================
REM Setup MCP Servers - Video Analysis AI
REM Uses shared venv from backend directory
REM ============================================

echo.
echo ========================================
echo   Video Analysis AI - MCP Servers Setup
echo ========================================
echo.

REM Get the absolute path to the project root
cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"
echo Project Root: %PROJECT_ROOT%
echo.

REM Check if backend venv exists
if not exist "%PROJECT_ROOT%\backend\venv\" (
    echo [ERROR] Backend virtual environment not found!
    echo [INFO] Please run setup-backend.bat first to create the shared venv.
    pause
    exit /b 1
)

echo [INFO] Using shared virtual environment from backend\venv
echo.

REM ============================================
REM 1. Transcription Agent
REM ============================================

echo ----------------------------------------
echo [1/3] Installing Transcription Agent dependencies...
echo ----------------------------------------

set "AGENT_DIR=%PROJECT_ROOT%\mcp-servers\transcription-agent"

if not exist "%AGENT_DIR%\" (
    echo [ERROR] Directory not found: %AGENT_DIR%
    echo [INFO] Please ensure all project files are present.
    pause
    exit /b 1
)

echo Working in: %AGENT_DIR%
echo [INFO] Installing dependencies...
pushd "%PROJECT_ROOT%\backend"
call venv\Scripts\activate.bat
cd /d "%AGENT_DIR%"
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    deactivate
    popd
    pause
    exit /b 1
)
deactivate
popd

echo [SUCCESS] Transcription Agent dependencies installed!
echo.

REM ============================================
REM 2. Vision Agent
REM ============================================

echo ----------------------------------------
echo [2/3] Installing Vision Agent dependencies...
echo ----------------------------------------

set "AGENT_DIR=%PROJECT_ROOT%\mcp-servers\vision-agent"

if not exist "%AGENT_DIR%\" (
    echo [ERROR] Directory not found: %AGENT_DIR%
    echo [INFO] Please ensure all project files are present.
    pause
    exit /b 1
)

echo Working in: %AGENT_DIR%
echo [INFO] Installing dependencies...
pushd "%PROJECT_ROOT%\backend"
call venv\Scripts\activate.bat
cd /d "%AGENT_DIR%"
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    deactivate
    popd
    pause
    exit /b 1
)
deactivate
popd

echo [SUCCESS] Vision Agent dependencies installed!
echo.

REM ============================================
REM 3. Report Agent
REM ============================================

echo ----------------------------------------
echo [3/3] Installing Report Agent dependencies...
echo ----------------------------------------

set "AGENT_DIR=%PROJECT_ROOT%\mcp-servers\report-agent"

if not exist "%AGENT_DIR%\" (
    echo [ERROR] Directory not found: %AGENT_DIR%
    echo [INFO] Please ensure all project files are present.
    pause
    exit /b 1
)

echo Working in: %AGENT_DIR%
echo [INFO] Installing dependencies...
pushd "%PROJECT_ROOT%\backend"
call venv\Scripts\activate.bat
cd /d "%AGENT_DIR%"
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    deactivate
    popd
    pause
    exit /b 1
)
deactivate
popd

echo [SUCCESS] Report Agent dependencies installed!
echo.

REM ============================================
REM Completion
REM ============================================

cd /d "%PROJECT_ROOT%"

echo ========================================
echo   All MCP Servers Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run "download-models.bat" to download Ollama models
echo   2. Run "start-mcp-servers.bat" to start all MCP servers
echo   3. Or run "start-all.bat" to start everything
echo.
echo For more information, see QUICKSTART.md
echo.

pause
