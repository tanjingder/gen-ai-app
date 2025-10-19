# 🚀 Video Analysis AI - Quick Start Guide# 🚀 Quick Start Guide# Video Analysis AI - Quick Start Guide



> **Last Updated:** October 19, 2025  

> **Local AI Desktop Application for Video Analysis**

**Updated:** October 18, 2025> **Local AI Desktop Application for Video Analysis**  

---

> Using Ollama, MCP Agents, React + Tauri

## 📋 Prerequisites

## Prerequisites

Before you begin, install these required software:

---

### Required

1. **Python 3.11+** - [Download](https://www.python.org/downloads/)

| Software | Version | Purpose | Download |

|----------|---------|---------|----------|2. **Node.js 18+** - [Download](https://nodejs.org/)## 🚀 Quick Start (5 Minutes)

| **Python** | 3.11+ | Backend runtime | [python.org](https://python.org) |

| **Node.js** | 18+ | Frontend build tools | [nodejs.org](https://nodejs.org) |3. **Rust** - [Download](https://rustup.rs/)

| **Rust** | Latest | Tauri desktop framework | [rustup.rs](https://rustup.rs) |

| **Ollama** | Latest | Local AI models | [ollama.ai](https://ollama.ai) |4. **Ollama** - [Download](https://ollama.ai/)### 1. Prerequisites



### Optional (but recommended)



| Software | Purpose | Download |## Step 1: Setup (First Time Only)Make sure you have installed:

|----------|---------|----------|

| **FFmpeg** | Audio extraction | [ffmpeg.org](https://ffmpeg.org) |- [Python 3.11+](https://python.org)

| **Git** | Version control | [git-scm.com](https://git-scm.com) |

```batch- [Node.js 18+](https://nodejs.org)

---

# Clone/download the project, then:- [Rust](https://rustup.rs)

## ⚡ Quick Start (3 Steps)

setup-all.bat- [Ollama](https://ollama.ai)

### Step 1: Setup Everything

```

```batch

# Run the setup script### 2. Setup (One Time)

setup.bat

**What this does:**

# Or directly:

scripts\setup-all.bat- Creates Python virtual environment```batch

```

- Installs all backend dependencies# Install all dependencies

**What happens:**

- ✅ Creates Python virtual environment- Installs MCP agent dependencies (shared venv).\setup-all.bat

- ✅ Installs backend dependencies (FastAPI, gRPC, Ollama client, etc.)

- ✅ Installs MCP agent dependencies (all 3 agents use shared venv)- Installs frontend dependencies (npm + Rust)

- ✅ Installs frontend dependencies (React, Tauri, TypeScript)

- ✅ Compiles protocol buffers# Download AI models



**Duration:** 5-10 minutes**Duration:** 5-10 minutes (depends on internet speed).\download-models.bat



---```



### Step 2: Download AI Models## Step 2: Install AI Models



```batch### 3. Start Application

# Download Ollama models

scripts\download-models.bat```batch

```

# Start Ollama (keep this running)```batch

**What happens:**

- ✅ Checks if Ollama is installedollama serve# Terminal 1: Start Ollama

- ✅ Starts Ollama server if not running

- ✅ Downloads llama3.2 (~4.7GB) - Main reasoning modelollama serve

- ✅ Downloads llava (~4.5GB) - Vision analysis model

# In another terminal, pull models:

**Duration:** 10-20 minutes (depends on internet speed)

ollama pull llama3.2# Terminal 2: Start everything

---

ollama pull llava.\start-all.bat

### Step 3: Start Application

``````

```batch

# Terminal 1: Start Ollama

ollama serve

**Duration:** 10-20 minutes (large downloads)**Done!** The desktop application will open.

# Terminal 2: Start application

start.bat



# Or directly:## Step 3: Start Application---

scripts\start-all.bat

```



**What happens:**```batch## 📖 Detailed Setup

- ✅ Backend gRPC server starts (port 50051)

- ✅ Desktop application launchesstart-all.bat

- ✅ MCP agents spawn automatically when needed

```### First Time Installation

**First launch:** May take 1-2 minutes to compile Rust code



---

**What this does:****Step 1: Clone and Navigate**

## 🎯 You're Done!

1. Opens backend server window (gRPC on port 50051)```batch

The application should now be running. Try:

1. **Chat with AI** - Type "Hello!" to test the connection2. Starts Tauri desktop applicationcd C:\Users\YourName\Project\gen-ai-app

2. **Upload a video** - Click "Upload Video" and select a video file

3. **Ask questions** - "What's in this video?" or "Summarize this"3. MCP agents spawn automatically as needed```

4. **Generate report** - "Create a PDF summary"



---

**First time:** May take 1-2 minutes to compile Rust code.**Step 2: Run Setup**

## 📦 What Gets Installed?

```batch

### Backend Dependencies (Python)

## That's It! 🎉.\setup-all.bat

**Core:**

- `grpcio` + `grpcio-tools` - gRPC server```

- `fastapi` - REST API (optional)

- `ollama` - Local LLM clientThe application should now be running. Try:This will:

- `mcp` - Model Context Protocol SDK

- Create Python virtual environment

**Video Processing:**

- `opencv-python` - Frame extraction1. **Upload a video** - Click "Upload Video" button- Install backend dependencies

- `ffmpeg-python` - Audio extraction

2. **Ask a question** - Type in the chat: "What's in this video?"- Install MCP agent dependencies

**Analysis Tools:**

- `pytesseract` - OCR text extraction3. **Generate report** - Request: "Create a PDF summary"- Install frontend npm packages

- `reportlab` - PDF generation

- `python-pptx` - PowerPoint generation- Compile protocol buffers



**Utilities:**---

- `pydantic` - Data validation

- `python-dotenv` - Environment config**Step 3: Download AI Models**

- `loguru` - Logging

## Daily Usage```batch

**Full list:** See `backend/requirements.txt`

.\download-models.bat

---

After initial setup, you only need:```

### Frontend Dependencies (Node.js)

Downloads:

**Core:**

- `react` + `react-dom` - UI framework```batch- `llama3.1` (~4.7GB) - Main AI model

- `typescript` - Type safety

- `@tauri-apps/api` - Desktop API# 1. Start Ollama (in one terminal)- `llava` (~4.5GB) - Vision model

- `@tauri-apps/cli` - Build tools

ollama serve

**UI & Styling:**

- `tailwindcss` - CSS framework**Step 4: Start Ollama**

- `lucide-react` - Icons

- `react-hot-toast` - Notifications# 2. Start app (in another terminal)



**Build Tools:**start-all.batOpen a **separate terminal** and keep it running:

- `vite` - Development server

- `@vitejs/plugin-react` - React support``````powershell



**Full list:** See `frontend/package.json`ollama serve



---## Stopping the Application```



### Rust Dependencies (Tauri)



**Core:**1. **Close the desktop app window** - This stops the frontend**Step 5: Start Everything**

- `tauri` - Desktop framework

- `serde` + `serde_json` - JSON serialization2. **Close the backend terminal** - This stops the gRPC server```batch

- `tokio` - Async runtime

3. **Stop Ollama** - Press Ctrl+C in the Ollama terminal.\start-all.bat

**gRPC Client:**

- `tonic` - gRPC framework```

- `prost` - Protocol buffers

- `tonic-build` - Code generation---



**Full list:** See `frontend/src-tauri/Cargo.toml`This opens:



---

## 📥 Download Whisper Model (Required for Transcription)

The transcription agent requires the Whisper model file, which is **too large for GitHub** (141 MB > 100 MB limit).

### Step-by-Step:

1. **Download the model:**
   - Visit: https://huggingface.co/ggerganov/whisper.cpp/tree/main
   - Click on `ggml-base.bin` (~142 MB)
   - Click the download button (↓) on the right side

2. **Create the models folder** (if it doesn't exist):
   ```powershell
   mkdir mcp-servers\transcription-agent\models
   ```

3. **Move the downloaded file:**
   - Move `ggml-base.bin` to: `mcp-servers\transcription-agent\models\`

4. **Verify the file:**
   ```powershell
   dir mcp-servers\transcription-agent\models\ggml-base.bin
   ```
   
   You should see: `ggml-base.bin` (approximately 141 MB)

### Alternative Models:

| Model | Size | Speed | Accuracy | Download |
|-------|------|-------|----------|----------|
| `ggml-tiny.bin` | 75 MB | ⚡⚡⚡ Fast | ⭐⭐ Basic | [Download](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-tiny.bin) |
| `ggml-base.bin` | 142 MB | ⚡⚡ Medium | ⭐⭐⭐ Good | [Download](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin) |
| `ggml-small.bin` | 466 MB | ⚡ Slower | ⭐⭐⭐⭐ Better | [Download](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-small.bin) |

> **Recommendation:** Use `ggml-base.bin` for the best balance of speed and accuracy.

---

## 📥 Download Whisper.cpp Executable (Required for Transcription)

The transcription agent also needs the `whisper.cpp` executable binaries, which are **not included in the repository**.

### Step-by-Step:

1. **Download the Windows release:**
   - Visit: https://github.com/ggerganov/whisper.cpp/releases
   - Look for the latest release
   - Download `whisper-bin-x64.zip` (or similar Windows binary package)

2. **Extract the ZIP file:**
   - Extract all contents to a temporary folder

3. **Copy to project:**
   ```powershell
   # Copy all files to whisper.cpp directory
   xcopy /E /I <extracted_folder>\* mcp-servers\transcription-agent\whisper.cpp\
   ```

4. **Verify installation:**
   ```powershell
   cd mcp-servers\transcription-agent
   .\whisper.cpp\main.exe --help
   ```
   
   You should see the help output from whisper.cpp.

### Required Files:

After extraction, you should have:
- `whisper.cpp/main.exe` - Main executable (used by the agent)
- `whisper.cpp/*.dll` - Required DLL files
- Other utility executables

> **Note:** These binaries are platform-specific (Windows only). Mac/Linux users need to build from source or download appropriate binaries.

> **See also:** `mcp-servers/transcription-agent/WHISPER_SETUP.md` for detailed setup instructions and troubleshooting.

---

## Troubleshooting- 3 MCP Agent windows (Transcription, Vision, Report)



### MCP Agent Dependencies- 1 Backend window (REST API + gRPC)



All three agents (Transcription, Vision, Report) share the backend virtual environment:### Application won't start- Desktop application



**Transcription Agent:**

- `mcp` - MCP SDK

- `ffmpeg-python` - Audio extraction**Check 1:** Is Ollama running?---

- Future: Whisper for transcription

```batch

**Vision Agent:**

- `mcp` - MCP SDK# Should see: Ollama is running## 🎯 Usage

- `opencv-python` - Frame extraction

- `ollama` - LLaVA for vision analysiscurl http://localhost:11434/

- `pytesseract` - OCR

- Future: YOLO for object detection```### Chat with AI



**Report Agent:**

- `mcp` - MCP SDK

- `reportlab` - PDF generation**Check 2:** Did setup complete?1. Open the desktop application

- `python-pptx` - PowerPoint generation

```batch2. Type your question: "Hello, how are you?"

---

# Should exist:3. Get real AI responses from Ollama!

## 📂 Project Structure After Setup

dir backend\venv

```

gen-ai-app/dir frontend\node_modules### Video Analysis (Future)

├── setup.bat               # Setup launcher

├── start.bat               # Start launcher```

│

├── scripts/                # All batch scripts1. Click "Upload Video"

│   ├── setup-all.bat

│   ├── start-all.bat**Fix:** Re-run `scripts\setup-all.bat`2. Select a video file

│   ├── download-models.bat

│   └── ...3. Ask questions about the video

│

├── backend/                # Python backend### Video upload fails4. Get AI-powered analysis

│   ├── venv/              ✨ Created during setup

│   ├── src/

│   │   ├── grpc_server/   # gRPC service

│   │   ├── mcp/           # Agent orchestrator**Check:** Is backend running?---

│   │   └── utils/         # Utilities

│   ├── main.py```batch

│   └── requirements.txt

│netstat -ano | findstr "50051"## 🔧 Troubleshooting

├── frontend/               # React + Tauri

│   ├── node_modules/      ✨ Created during setup# Should show: LISTENING

│   ├── src/               # React code

│   ├── src-tauri/         # Rust code```### "Backend not reachable"

│   ├── package.json

│   └── Cargo.toml```batch

│

├── mcp-servers/            # MCP agents**Fix:** Start backend: `scripts\start-backend.bat`# Check if backend is running

│   ├── transcription-agent/

│   ├── vision-agent/# If not, start it:

│   └── report-agent/

│### Chat not responding.\start-backend.bat

├── proto/                  # gRPC definitions

│   └── video_analysis.proto```

│

### "Transcription model not found" Error

**Symptom:** Transcription fails with "Model not found" error

**Cause:** The `ggml-base.bin` model file is missing (not included in GitHub due to 100MB limit)

**Fix:**
1. Download the model from HuggingFace: https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin
2. Place it in: `mcp-servers\transcription-agent\models\ggml-base.bin`
3. Verify: `dir mcp-servers\transcription-agent\models\ggml-base.bin`

See the "📥 Download Whisper Model" section above for detailed instructions.

### "Whisper.cpp not found" Error

**Symptom:** Transcription fails with "whisper.cpp executable not found" error

**Cause:** The `whisper.cpp` binaries are missing (not included in GitHub - third-party binaries)

**Fix:**
1. Download Windows binaries: https://github.com/ggerganov/whisper.cpp/releases
2. Extract `whisper-bin-x64.zip`
3. Copy contents to: `mcp-servers\transcription-agent\whisper.cpp\`
4. Verify: `mcp-servers\transcription-agent\whisper.cpp\main.exe --help`

See the "📥 Download Whisper.cpp Executable" section above for detailed instructions.

│

└── data/                   ✨ Created at runtime**Check:** Ollama models installed?

    └── sessions/

        └── {session_id}/```batch### "Ollama connection failed"

            ├── uploads/    # Video files

            ├── cache/      # Analysis cacheollama list```powershell

            ├── temp/       # Temporary files

            └── reports/    # Generated PDFs/PPTs# Should show: llama3.2 and llava# Make sure Ollama is running:

```

```ollama serve

---

```

## 🔧 Daily Usage

**Fix:** Pull models (see Step 2 above)

After initial setup, you only need:

### "Model not found"

```batch

# Terminal 1: Start Ollama (keep running)### "Module not found" errors```powershell

ollama serve

# Download the model:

# Terminal 2: Start application

start.bat**Fix:** Reinstall dependenciesollama pull llama3.1

```

```batch```

That's it! The application will:

- Start backend gRPC servercd backend

- Launch desktop app

- Spawn MCP agents automatically when neededcall venv\Scripts\activate.bat### Frontend won't start



---pip install -r requirements.txt```batch



## 🛠️ Troubleshooting```# Install dependencies:



### ❌ "Backend not reachable"cd frontend



**Problem:** Backend gRPC server not running---npm install



**Solution:**```

```batch

# Check if backend is running## Project Structure

netstat -ano | findstr "50051"

### Backend errors

# If not running, start it

scripts\start-backend.bat``````batch

```

gen-ai-app/# Reinstall backend:

---

├── start-all.bat          ⭐ Start here!.\setup-backend.bat

### ❌ "Ollama connection failed"

├── setup-all.bat          🔧 Run once for setup```

**Problem:** Ollama server not running

├── backend/               🐍 Python gRPC server

**Solution:**

```batch│   ├── main.py           → Entry point---

# Start Ollama

ollama serve│   ├── venv/             → Virtual environment



# In another terminal, check if working│   └── uploads/          → Uploaded videos go here## 📂 Project Structure

ollama list

```├── mcp-servers/          🤖 AI agents (spawn on-demand)



---│   ├── transcription-agent/```



### ❌ "Model not found: llama3.2"│   ├── vision-agent/gen-ai-app/



**Problem:** AI models not downloaded│   └── report-agent/├── backend/              # Python backend



**Solution:**└── frontend/             ⚛️ React + Tauri app│   ├── main.py          # Entry point (REST + gRPC)

```batch

# Download models    ├── src/              → React code│   ├── rest_api.py      # REST API server

scripts\download-models.bat

    └── src-tauri/        → Rust gRPC client│   └── src/             # Source code

# Or manually

ollama pull llama3.2```├── frontend/            # React + Tauri app

ollama pull llava

```│   ├── src/             # React components



------│   └── src-tauri/       # Rust/Tauri code



### ❌ "Virtual environment not found"├── mcp-servers/         # MCP agents



**Problem:** Setup didn't complete successfully## Advanced Usage│   ├── transcription-agent/



**Solution:**│   ├── vision-agent/

```batch

# Re-run setup### Manual Backend Start│   └── report-agent/

scripts\setup-all.bat

├── proto/               # Protocol buffers

# Or setup individually

scripts\setup-backend.bat```batch├── docs/                # Detailed documentation

scripts\setup-mcp-servers.bat

scripts\setup-frontend.batcd backend├── setup-all.bat       # Setup everything

```

call venv\Scripts\activate.bat├── start-all.bat       # Start everything

---

python main.py└── download-models.bat # Download AI models

### ❌ "FFmpeg not found"

``````

**Problem:** FFmpeg not installed or not in PATH



**Solution:**

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)### Manual Frontend Start---

2. Extract to a folder (e.g., `C:\ffmpeg`)

3. Add to PATH: `C:\ffmpeg\bin`

4. Restart terminal and test: `ffmpeg -version`

```batch## 🎮 Available Scripts

---

cd frontend

### ❌ Frontend won't start

npm run tauri:dev### Setup Scripts

**Problem:** Node modules corrupted or Rust not installed

```- `scripts\setup-all.bat` - Setup everything (recommended)

**Solution:**

```batch- `setup-backend.bat` - Setup backend only

# Check Rust

rustc --version### Test MCP Agents- `setup-frontend.bat` - Setup frontend only



# If not installed, get from https://rustup.rs- `setup-mcp-servers.bat` - Setup MCP agents only



# Reinstall frontend```batch

cd frontend

Remove-Item -Recurse -Force node_modulescd backend### Start Scripts

npm install

```python test_mcp_stdio.py- `scripts\start-all.bat` - Start everything (recommended)



---```- `scripts\start-backend.bat` - Start backend only



### ❌ Slow first startup- `start-mcp-servers.bat` - Start MCP agents only



**Reason:** Rust code compiling for first time (normal!)### View Logs



**Expected:** 1-2 minutes for initial Tauri build### Utility Scripts



**Future starts:** Much faster (< 10 seconds)```batch- `download-models.bat` - Download Ollama models



---# Backend logs- `compile-proto.bat` - Compile protocol buffers



### ❌ "Port 50051 already in use"type backend\app.log



**Problem:** Another backend instance running---



**Solution:**# Uploaded videos

```batch

# Find and kill the processdir backend\uploads## ⚙️ Configuration

netstat -ano | findstr "50051"

# Note the PID, then:

taskkill /PID <pid> /F

# Generated reportsEdit `backend/.env` to customize:

# Or restart your computer

```dir backend\reports



---``````env



## 💡 Tips & Best Practices# Ollama



### Performance---OLLAMA_HOST=http://localhost:11434



1. **Keep Ollama running** - Avoid restart overheadOLLAMA_MODEL=llama3.1

2. **Use cache** - Repeated queries are instant

3. **Close unused sessions** - Free up memory## Getting Help

4. **GPU acceleration** - Ollama uses GPU if available

# Ports

### Storage

1. **Check logs:** `backend/app.log`GRPC_PORT=50051

```

Video file: ~50-200MB2. **Read docs:** 

Extracted frames: ~2-10MB

Analysis cache: ~1-5MB per video   - `README.md` - Full documentation# MCP Agents

Reports: ~100KB-1MB each

   - `ARCHITECTURE_UPDATE.md` - Technical detailsTRANSCRIPTION_MCP_PORT=8001

Recommendation: 10GB+ free space for video analysis

```   - `CLEANUP_SUMMARY.md` - Recent changesVISION_MCP_PORT=8002



### Security3. **Common issues:** See Troubleshooting section aboveREPORT_MCP_PORT=8003



- ✅ All processing is local (no cloud)```

- ✅ No telemetry or tracking

- ✅ Videos never leave your machine---

- ✅ Private and offline-capable

---

---

## What's New (Oct 2025)

## 📚 Next Steps

## 🏗️ Architecture

### Learn More

✅ **Simpler startup** - No more MCP server windows  

- **Architecture:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details

- **README:** See [README.md](README.md) for full feature list✅ **Faster upload** - Native gRPC (no REST API)  ```

- **Scripts:** See [scripts/README.md](scripts/README.md) for script documentation

✅ **Better performance** - Agents spawn on-demand  Frontend (Tauri Desktop App)

### Try Features

✅ **Type safety** - Rust gRPC client with protobuf           ↓ HTTP/Tauri Invoke

1. **Upload a video** - Drag and drop or click to browse

2. **Ask questions** - Natural language queries about video contentBackend Server (Python)

3. **Generate reports** - Request PDF or PowerPoint summaries

4. **View history** - Switch between sessionsSee `ARCHITECTURE_UPDATE.md` for technical details.  ├─ REST API (port 8000)

5. **Download reports** - Save PDFs and PPTs locally

  └─ gRPC (port 50051)

### Advanced

---         ↓

```batch

# Compile protocol buffers after changes    Ollama AI (llama3.1)

scripts\compile-proto.bat

**Enjoy your fully local AI video analyzer!** 🎬🤖         ↓

# Setup individual components

scripts\setup-backend.batMCP Agent Servers

scripts\setup-frontend.bat  ├─ Transcription (port 8001)

scripts\setup-mcp-servers.bat  ├─ Vision (port 8002)

  └─ Report (port 8003)

# Start only backend (for development)```

scripts\start-backend.bat

```**Communication Flow:**

1. User types in frontend

---2. Frontend calls Tauri command

3. Tauri calls backend REST API

## 🎉 Success!4. Backend calls Ollama

5. Ollama returns AI response

You now have a fully functional local AI video analysis desktop application!6. Response flows back to user



**What you can do:**---

- ✅ Analyze videos completely offline

- ✅ Chat with AI about video content  ## 📚 Learn More

- ✅ Generate professional reports

- ✅ Keep all data private and local- `INTEGRATION_COMPLETE.md` - Detailed integration guide

- `OLLAMA_INTEGRATION.md` - Ollama setup details

**Need help?** Check the troubleshooting section above or open an issue.- `docs/ARCHITECTURE.md` - Architecture details

- `docs/DIAGRAMS.md` - System diagrams

---- `docs/ROADMAP.md` - Future plans



**Enjoy your privacy-focused AI assistant!** 🚀---



> Built with ❤️ using Ollama, MCP, React, and Tauri## 🎓 How It Works


### 1. Backend Integration
- Single Python process runs both REST API and gRPC
- REST API provides easy access for frontend
- gRPC provides advanced features for future

### 2. AI Processing
- Ollama runs locally (no cloud, fully private)
- llama3.1 model handles conversations
- llava model handles image/video analysis

### 3. MCP Agents
- Modular agents for different tasks
- Transcription: Audio extraction and speech-to-text
- Vision: Frame analysis and object detection
- Report: PDF and PowerPoint generation

### 4. Frontend
- Tauri provides native desktop experience
- React provides modern UI
- TypeScript ensures type safety

---

## ✨ Features

**Current:**
- ✅ Real AI conversations with Ollama
- ✅ Desktop application
- ✅ Session management
- ✅ Error handling
- ✅ Local processing (privacy)

**Planned:**
- 🔄 Video upload and analysis
- 🔄 Transcription and timestamps
- 🔄 Visual analysis and OCR
- 🔄 Report generation (PDF/PPT)
- 🔄 Multi-video comparison

---

## 🆘 Support

**Common Issues:**

1. **Port already in use**
   - Close other instances
   - Restart your computer

2. **Slow first startup**
   - First time compiles Rust (1-2 min)
   - Normal behavior

3. **Missing dependencies**
   - Run `scripts\setup-all.bat` again
   - Check Python/Node/Rust installed

4. **Ollama errors**
   - Make sure `ollama serve` is running
   - Pull models: `ollama pull llama3.1`

---

## 🎉 Success!

You now have a fully functional local AI desktop application!

**What you can do:**
- Chat with AI (no internet needed after setup)
- Analyze videos (coming soon)
- Generate reports (coming soon)
- Everything runs locally on your machine

**Enjoy your privacy-focused AI assistant!** 🚀

---

## 📝 License

This project is for personal and educational use.

## 🤝 Contributing

This is a personal project. Feel free to fork and customize!

---

**Last Updated:** October 18, 2025
