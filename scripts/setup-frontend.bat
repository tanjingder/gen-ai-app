@echo off
REM ============================================
REM Setup Frontend - Video Analysis AI
REM ============================================

echo.
echo ========================================
echo   Video Analysis AI - Frontend Setup
echo ========================================
echo.

cd /d "%~dp0..\frontend"

REM Check if node_modules exists
if exist "node_modules\" (
    echo [INFO] node_modules already exists. Skipping installation.
    echo [INFO] If you want to reinstall, delete node_modules folder first.
) else (
    echo [1/2] Installing npm dependencies...
    echo [INFO] This may take a few minutes...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install npm dependencies
        pause
        exit /b 1
    )
    echo [SUCCESS] npm dependencies installed.
)

REM Check Rust/Tauri setup
echo.
echo [2/2] Checking Tauri setup...
rustc --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Rust is not installed or not in PATH.
    echo [WARNING] Tauri requires Rust. Please install from https://rustup.rs/
    echo.
) else (
    echo [SUCCESS] Rust is installed.
)

echo.
echo ========================================
echo   Frontend Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure backend is running
echo 2. Run start-frontend.bat to start the application
echo.
pause
