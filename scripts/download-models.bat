@echo off
REM ============================================
REM Download Ollama Models - Video Analysis AI
REM ============================================

echo.
echo ========================================
echo   Downloading Ollama Models
echo ========================================
echo.

REM Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not in PATH!
    echo.
    echo Please install Ollama:
    echo 1. Visit https://ollama.ai
    echo 2. Download and install Ollama for Windows
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

echo [INFO] Ollama is installed.
echo.

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama server is not running.
    echo [INFO] Starting Ollama server in a new window...
    start "Ollama Server" cmd /k "ollama serve"
    echo [INFO] Waiting for Ollama to start...
    timeout /t 5 >nul
)

REM Pull llama3.2
echo.
echo [1/2] Pulling llama3.2 model...
echo [INFO] This may take several minutes depending on your internet speed...
ollama pull llama3.2
if errorlevel 1 (
    echo [ERROR] Failed to pull llama3.2 model
    pause
    exit /b 1
)
echo [SUCCESS] llama3.2 model downloaded.

REM Pull llava
echo.
echo [2/2] Pulling llava model...
echo [INFO] This may take several minutes depending on your internet speed...
ollama pull llava
if errorlevel 1 (
    echo [ERROR] Failed to pull llava model
    pause
    exit /b 1
)
echo [SUCCESS] llava model downloaded.

REM Verify models
echo.
echo [INFO] Verifying installed models...
ollama list

echo.
echo ========================================
echo   All Models Downloaded Successfully!
echo ========================================
echo.
echo Models installed:
echo - llama3.2 (Main reasoning model)
echo - llava (Vision analysis model)
echo.
echo You can now run the application!
echo.
pause
