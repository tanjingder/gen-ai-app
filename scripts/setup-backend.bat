@echo off
REM ============================================
REM Setup Backend - Video Analysis AI
REM ============================================

echo.
echo ========================================
echo   Video Analysis AI - Backend Setup
echo ========================================
echo.

cd /d "%~dp0..\backend"

REM Check if venv exists
if exist "venv\" (
    echo [INFO] Virtual environment already exists.
) else (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

REM Activate virtual environment and install dependencies
echo.
echo [2/4] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed.

REM Create .env if it doesn't exist
echo.
echo [3/4] Checking environment configuration...
if exist ".env" (
    echo [INFO] .env file already exists.
) else (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env
    echo [SUCCESS] .env file created.
)

REM Compile proto files
echo.
echo [4/4] Compiling protocol buffer definitions...
cd /d "%~dp0.."
python -m grpc_tools.protoc ^
    --proto_path="proto" ^
    --python_out="backend\src\grpc_server" ^
    --grpc_python_out="backend\src\grpc_server" ^
    "proto\video_analysis.proto"

if errorlevel 1 (
    echo [ERROR] Failed to compile proto files
    pause
    exit /b 1
)

echo [INFO] Fixing imports in generated files...
python -c "import pathlib; f = pathlib.Path('backend/src/grpc_server/video_analysis_pb2_grpc.py'); content = f.read_text(); content = content.replace('import video_analysis_pb2 as video__analysis__pb2', 'from . import video_analysis_pb2 as video__analysis__pb2'); f.write_text(content)" 2>nul

echo [SUCCESS] Proto files compiled.

cd /d "%~dp0..\backend"
deactivate

echo.
echo ========================================
echo   Backend Setup Complete!
echo ========================================
echo.
echo Generated files:
echo - src\grpc_server\video_analysis_pb2.py
echo - src\grpc_server\video_analysis_pb2_grpc.py
echo.
echo Next steps:
echo 1. Make sure Ollama is running: ollama serve
echo 2. Run setup-mcp-servers.bat to setup MCP servers
echo 3. Run start-backend.bat to start the backend
echo.
pause
