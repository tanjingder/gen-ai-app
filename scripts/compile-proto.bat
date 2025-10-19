@echo off
REM ============================================
REM Compile Proto Files - Video Analysis AI
REM ============================================

echo.
echo ========================================
echo   Compiling Protocol Buffer Definitions
echo ========================================
echo.

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

REM Check if backend venv exists
if not exist "%PROJECT_ROOT%\backend\venv\" (
    echo [ERROR] Backend virtual environment not found!
    echo [INFO] Please run setup-backend.bat first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
cd /d "%PROJECT_ROOT%\backend"
call venv\Scripts\activate.bat

echo [INFO] Compiling proto files...
echo.

python -m grpc_tools.protoc ^
    --proto_path="%PROJECT_ROOT%\proto" ^
    --python_out="%PROJECT_ROOT%\backend\src\grpc_server" ^
    --grpc_python_out="%PROJECT_ROOT%\backend\src\grpc_server" ^
    "%PROJECT_ROOT%\proto\video_analysis.proto"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to compile proto files
    deactivate
    pause
    exit /b 1
)

echo [INFO] Fixing imports in generated files...
python -c "import pathlib; f = pathlib.Path('backend/src/grpc_server/video_analysis_pb2_grpc.py'); content = f.read_text(); content = content.replace('import video_analysis_pb2 as video__analysis__pb2', 'from . import video_analysis_pb2 as video__analysis__pb2'); f.write_text(content)" 2>nul

echo.
echo [SUCCESS] Proto files compiled successfully!
echo.
echo Generated files:
echo - backend\src\grpc_server\video_analysis_pb2.py
echo - backend\src\grpc_server\video_analysis_pb2_grpc.py
echo.

deactivate

pause
