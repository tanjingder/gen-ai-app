@echo off
REM Test Transcription Agent
REM This script tests the agent functionality

echo ============================================
echo Testing Transcription Agent
echo ============================================
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "..\..\backend\venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup-backend.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call "..\..\backend\venv\Scripts\activate.bat"

REM Run test
python test_agent.py

echo.
echo ============================================
echo Test complete!
echo ============================================
pause
