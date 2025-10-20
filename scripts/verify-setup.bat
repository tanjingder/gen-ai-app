@echo off
REM Verification script for Video Analysis AI prerequisites
REM Checks if all required tools are properly installed

title Video Analysis AI - Setup Verification

cd /d "%~dp0.."

echo ===============================================
echo   Video Analysis AI - Setup Verification
echo ===============================================
echo.

REM Initialize counters and failed items list
set PASS_COUNT=0
set FAIL_COUNT=0
set "FAILED_ITEMS="

REM ===========================================
REM Core Requirements Check
REM ===========================================
echo [CORE REQUIREMENTS]
echo.

REM Check Git
echo [1/6] Checking Git...
git --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Git is installed
    git --version
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Git is NOT installed
    echo        Download: https://git-scm.com/downloads/win
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Git; "
)
echo.

REM Check Python
echo [2/6] Checking Python...
python --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Python is installed
    python --version
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Python is NOT installed
    echo        Download: Microsoft Store (Python 3.11+)
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Python 3.11+; "
)
echo.

REM Check Node.js
echo [3/6] Checking Node.js...
node --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Node.js is installed
    node --version
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Node.js is NOT installed
    echo        Download: https://nodejs.org/en/download
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Node.js 18+; "
)
echo.

REM Check Rust
echo [4/6] Checking Rust...
rustc --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Rust is installed
    rustc --version
    cargo --version >nul 2>&1
    if %errorlevel%==0 (
        cargo --version
    )
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Rust is NOT installed
    echo        Download: https://rustup.rs/
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Rust; "
)
echo.

REM Check Ollama
echo [5/6] Checking Ollama...
curl -s http://localhost:11434/ >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Ollama is running
    
    REM Check for required models
    ollama list | findstr "llama3.2" >nul 2>&1
    if %errorlevel%==0 (
        echo        [OK] llama3.2 model is installed
    ) else (
        echo        [WARNING] llama3.2 model NOT found
        echo        Run: ollama pull llama3.2
    )
    
    ollama list | findstr "llava" >nul 2>&1
    if %errorlevel%==0 (
        echo        [OK] llava model is installed
    ) else (
        echo        [WARNING] llava model NOT found
        echo        Run: ollama pull llava
    )
    set /a PASS_COUNT+=1
) else (
    echo [WARNING] Ollama is not running or not installed
    echo        Download: https://ollama.com/download/windows
    echo        Start with: ollama serve
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Ollama; "
)
echo.

@REM REM Check Visual Studio C++ Build Tools
@REM echo [6/6] Checking Visual Studio C++ Build Tools...
@REM where cl.exe >nul 2>&1
@REM if %errorlevel%==0 (
@REM     echo [PASS] Visual Studio C++ tools are available
@REM     set /a PASS_COUNT+=1
@REM ) else (
@REM     REM Try to find VS installation
@REM     if exist "C:\Program Files\Microsoft Visual Studio\2022" (
@REM         echo [WARNING] Visual Studio 2022 detected but cl.exe not in PATH
@REM         echo        Open "Developer Command Prompt for VS 2022" or
@REM         echo        Make sure "Desktop development with C++" is installed
@REM         set /a FAIL_COUNT+=1
@REM         set "FAILED_ITEMS=%FAILED_ITEMS%- Visual Studio C++ Build Tools; "
@REM     ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
@REM         echo [WARNING] Visual Studio 2019 detected but cl.exe not in PATH
@REM         echo        Open "Developer Command Prompt for VS 2019" or
@REM         echo        Make sure "Desktop development with C++" is installed
@REM         set /a FAIL_COUNT+=1
@REM         set "FAILED_ITEMS=%FAILED_ITEMS%- Visual Studio C++ Build Tools; "
@REM     ) else (
@REM         echo [FAIL] Visual Studio C++ Build Tools NOT found
@REM         echo        Download: https://visualstudio.microsoft.com/
@REM         echo        Install "Desktop development with C++" workload
@REM         set /a FAIL_COUNT+=1
@REM         set "FAILED_ITEMS=%FAILED_ITEMS%- Visual Studio C++ Build Tools; "
@REM     )
@REM )
@REM echo.

REM ===========================================
REM Agent Tools Check
REM ===========================================
echo [AGENT TOOLS]
echo.

REM Check FFmpeg
echo [1/5] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] FFmpeg is installed
    ffmpeg -version | findstr "ffmpeg version"
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] FFmpeg is NOT in PATH
    echo        Download: https://www.gyan.dev/ffmpeg/builds/
    echo        Extract and add bin folder to System Path
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- FFmpeg; "
)
echo.

REM Check Tesseract
echo [2/5] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Tesseract OCR is installed
    tesseract --version | findstr "tesseract"
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Tesseract OCR is NOT in PATH
    echo        Download: https://sourceforge.net/projects/tesseract-ocr.mirror/files/5.5.0/
    echo        Install and add to System Path
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Tesseract OCR; "
)
echo.

REM Check Protoc
echo [3/5] Checking Protocol Buffers Compiler (protoc)...
protoc --version >nul 2>&1
if %errorlevel%==0 (
    echo [PASS] Protoc is installed
    protoc --version
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] Protoc is NOT in PATH
    echo        Download: https://github.com/protocolbuffers/protobuf/releases
    echo        Extract and add bin folder to System Path
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Protoc; "
)
echo.

REM Check Whisper model
echo [4/5] Checking Whisper model (ggml-base.bin)...
if exist "mcp-servers\transcription-agent\models\ggml-base.bin" (
    echo [PASS] ggml-base.bin model found
    echo        Size: ~142 MB
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] ggml-base.bin model NOT found
    echo        Download: https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin
    echo        Place in: mcp-servers\transcription-agent\models\ggml-base.bin
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- Whisper model; "
)
echo.

REM Check Whisper.cpp executable
echo [5/5] Checking whisper.cpp executable...
if exist "mcp-servers\transcription-agent\whisper.cpp\main.exe" (
    echo [PASS] whisper.cpp executable found
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] whisper.cpp executable NOT found
    echo        Download: https://github.com/ggerganov/whisper.cpp/releases
    echo        Extract whisper-bin-x64.zip to: mcp-servers\transcription-agent\whisper.cpp\
    set /a FAIL_COUNT+=1
    set "FAILED_ITEMS=%FAILED_ITEMS%- whisper.cpp; "
)
echo.

REM ===========================================
REM Summary
REM ===========================================
echo ===============================================
echo   VERIFICATION SUMMARY
echo ===============================================
echo.
echo Passed: %PASS_COUNT%
echo Failed: %FAIL_COUNT%
echo.

if %FAIL_COUNT%==0 (
    echo [SUCCESS] All prerequisites are installed!
    echo.
    echo Next steps:
    echo   1. Run: scripts\setup-all.bat
    echo   2. Run: scripts\start-all.bat
    echo.
) else (
    echo [ACTION REQUIRED] Please install missing prerequisites
    echo.
    echo Failed items:
    echo %FAILED_ITEMS%
    echo.
    echo See installation links above or check:
    echo   - README.md for detailed instructions
    echo   - QUICKSTART.md for setup guide
    echo.
)

echo Press any key to exit...
pause >nul



