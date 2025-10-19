# Local AI Video Analysis Desktop Application

A fully offline AI desktop application for analyzing and querying short video files with multi-agent architecture.

## � Latest Update (Oct 2025)

✅ **Native gRPC Architecture** - Frontend now connects directly via gRPC (no REST API)  
✅ **Stdio MCP Agents** - Agents spawn on-demand via stdio (no HTTP servers)  
✅ **Simplified Deployment** - One backend process, agents spawn as needed  
✅ **Type-Safe Communication** - Rust gRPC client with protobuf definitions  

See [ARCHITECTURE_UPDATE.md](./ARCHITECTURE_UPDATE.md) for details.

## �🏗️ Architecture

```
[Frontend: React + Tauri Desktop App]
        ↓
    Tauri Rust Backend
    (Full gRPC Client)
        ↓ gRPC (port 50051)
        ↓ Binary Protocol
        ↓
[Python Backend: gRPC Server]
        ↓
    MCP Orchestrator
    (stdio-based)
        ↓
    🧠 Ollama (Planner + Summarizer)
        ↓
    JSON Plan Output
        ↓
│ MCP Agents (Spawn On-Demand via stdio)          
│--------------------------------------------
│ 🎧 Transcription Agent (Speech-to-text)
│ 👁️ Vision Agent (Object detection, OCR)
│ 📝 Report Agent (PDF/PPT generation)
        ↓
[Final Output → PDF/PPT Summary]
```

## 📁 Project Structure

```
gen-ai-app/
├── backend/                    # Python backend service
│   ├── src/
│   │   ├── agents/            # Agent implementations
│   │   ├── grpc_server/       # gRPC server implementation
│   │   ├── mcp/               # MCP client/orchestrator
│   │   ├── models/            # Data models
│   │   └── utils/             # Utility functions
│   ├── requirements.txt
│   └── main.py
│
├── mcp-servers/               # Model Context Protocol Servers
│   ├── transcription-agent/   # Audio transcription MCP server
│   ├── vision-agent/          # Vision analysis MCP server
│   └── report-agent/          # Report generation MCP server
│
├── frontend/                  # React + Tauri frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # gRPC client
│   │   └── App.tsx
│   └── src-tauri/            # Tauri backend
│
├── proto/                     # gRPC protobuf definitions
│   └── video_analysis.proto
│
└── docs/                      # Documentation
```

## 🚀 Features

- **Fully Offline**: All AI processing runs locally using Ollama
- **Multi-Agent System**: Specialized agents for transcription, vision, and reporting
- **MCP Architecture**: Modular agent design using Model Context Protocol
- **Desktop Native**: Built with Tauri for lightweight, secure desktop experience
- **Chat Interface**: Conversational UI for natural interaction
- **Video Analysis**: Extract and query information from video files
- **Report Generation**: Automated PDF/PPT report creation

## 🛠️ Tech Stack

### Frontend
- **React** - UI framework
- **Tauri** - Desktop application framework
- **TypeScript** - Type-safe development
- **Rust (tonic + prost)** - Native gRPC client

### Backend
- **Python 3.11+** - Backend runtime
- **gRPC** - High-performance RPC (port 50051)
- **Ollama** - Local LLM inference
- **MCP SDK 0.9.1** - Agent communication via stdio

### MCP Agents (On-Demand Stdio)
- **Transcription**: Audio extraction and transcription
- **Vision**: Frame analysis, object detection, OCR
- **Report**: PDF/PPT generation with reportlab and python-pptx

## 📋 Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm
- Rust (for Tauri)
- Ollama installed and running
- At least 16GB RAM recommended

## 🔧 Quick Setup (Automated)

### Windows - One Command Setup

```batch
setup.bat
# Or directly: scripts\setup-all.bat
```

This will:
1. Set up Python backend with shared virtual environment
2. Install all MCP agent dependencies (using shared venv)
3. Set up Tauri frontend (Rust + Node.js dependencies)

### Starting the Application

```batch
start.bat
# Or directly: scripts\start-all.bat
```

This will:
1. Start backend (gRPC server on port 50051)
2. Start frontend (Tauri desktop application)

**Note:** MCP agents spawn automatically on-demand. No separate server processes needed!

---

## 🔧 Manual Setup (If Needed)

### 1. Install Ollama

```powershell
# Download and install from https://ollama.ai
# Pull required models
ollama serve
ollama pull llama3.2
ollama pull llava  # For vision tasks
```

### 2. Backend Setup

```batch
cd backend
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 3. MCP Agents Setup

**Important:** MCP agents now use the shared backend venv!

```batch
cd backend
call venv\Scripts\activate.bat
cd ..\mcp-servers\transcription-agent
pip install -r requirements.txt
cd ..\vision-agent
pip install -r requirements.txt
cd ..\report-agent
pip install -r requirements.txt
```

### 4. Frontend Setup

```batch
cd frontend
npm install
cargo build --manifest-path=src-tauri/Cargo.toml
```

### 5. Start Backend

```batch
scripts\start-backend.bat
# Or manually:
cd backend
call venv\Scripts\activate.bat
python main.py
```

### 6. Start Frontend

```batch
cd frontend
npm run tauri:dev
```

## 🎯 Usage Flow

1. **Launch Application**: Start the Tauri desktop app
2. **Chat with AI**: Interact with Ollama through the chat interface
3. **Upload Video**: Select a video file for analysis
4. **Ask Questions**: Query the video content naturally
5. **Get Reports**: Request PDF/PPT summaries of the analysis

## 🔌 MCP Server Integration

Each agent is implemented as an MCP server with specific tools:

### Transcription Agent
- `extract_audio` - Extract audio from video
- `transcribe_audio` - Convert speech to text
- `get_timestamps` - Get timestamped transcription

### Vision Agent
- `extract_frames` - Extract key frames from video
- `analyze_frame` - Describe frame content
- `detect_objects` - Identify objects in frames
- `extract_text` - OCR text from frames
- `analyze_charts` - Extract data from graphs/charts

### Report Agent
- `create_pdf_report` - Generate PDF summary
- `create_ppt_report` - Generate PowerPoint presentation
- `format_content` - Structure analyzed content

## 🧠 Agent Orchestration

The backend uses Ollama as the planning LLM to:
1. Understand user intent
2. Determine which agents to invoke
3. Coordinate agent responses
4. Synthesize final answers

## 📝 Development Notes

- All processing is local - no cloud APIs
- MCP servers communicate via stdio or HTTP
- gRPC used for frontend-backend communication
- Video files stored temporarily during processing
- Models cached locally for fast inference

## 🔐 Security

- No external network calls during video processing
- Local file system access only
- Desktop sandboxing via Tauri

## � Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Complete setup and usage instructions
- **[Architecture](docs/ARCHITECTURE.md)** - Software architecture and design patterns
- **[Scripts Documentation](scripts/README.md)** - All batch script references

## �📄 License

MIT License

## 🤝 Contributing

This is a local development project. Feel free to adapt and extend.
