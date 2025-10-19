# Backend Setup Guide

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running
- ffmpeg (for audio extraction)

## Installation

1. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Create environment configuration:
```powershell
cp .env.example .env
```

Edit `.env` and configure your settings.

## Running

Start the backend server:

```powershell
python main.py
```

The gRPC server will start on `localhost:50051` by default.

## Configuration

Key configuration options in `.env`:

- `GRPC_PORT`: gRPC server port (default: 50051)
- `OLLAMA_HOST`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Main LLM model (default: llama3.2)
- `OLLAMA_VISION_MODEL`: Vision model (default: llava)
- `MAX_VIDEO_SIZE_MB`: Maximum video file size (default: 500)
- `MAX_VIDEO_DURATION_SECONDS`: Maximum video duration (default: 600)

## Project Structure

```
backend/
├── src/
│   ├── agents/           # Agent implementations (future)
│   ├── grpc_server/      # gRPC server
│   │   ├── server.py     # Server setup
│   │   └── service.py    # Service implementation
│   ├── mcp/              # MCP client and orchestrator
│   │   ├── client.py     # MCP client
│   │   └── orchestrator.py  # Agent coordination
│   ├── models/           # Data models
│   │   └── video_store.py
│   └── utils/            # Utilities
│       ├── config.py     # Configuration
│       └── setup.py      # Setup functions
├── requirements.txt
└── main.py              # Entry point
```

## Dependencies

Core dependencies:
- `fastapi` - Web framework
- `grpcio` - gRPC framework
- `ollama` - Ollama Python client
- `mcp` - Model Context Protocol SDK
- `opencv-python` - Video processing
- `loguru` - Logging

## MCP Integration

The backend orchestrates three MCP servers:
1. **Transcription Agent** (port 8001)
2. **Vision Agent** (port 8002)
3. **Report Agent** (port 8003)

Make sure all MCP servers are running before starting the backend.

## Troubleshooting

### Ollama Connection Failed

Ensure Ollama is running:
```powershell
ollama serve
```

Pull required models:
```powershell
ollama pull llama3.2
ollama pull llava
```

### Proto Compilation Errors

Proto files are compiled automatically on first run. If you encounter errors:
1. Check that `proto/video_analysis.proto` exists
2. Ensure `grpcio-tools` is installed
3. Delete generated `*_pb2.py` files and restart

### Import Errors

Make sure you're in the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

## Development

### Adding New Agents

1. Create agent implementation in `src/agents/`
2. Register tools in MCP server
3. Update orchestrator to call new agent
4. Update system prompt with new capabilities

### Logging

Logs are written to console and `app.log` by default. Configure in `.env`:
```
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## API Documentation

See `proto/video_analysis.proto` for the complete gRPC API definition.

Main services:
- `Chat` - Streaming chat interface
- `UploadVideo` - Upload video files
- `QueryVideo` - Query video content
- `GenerateReport` - Generate PDF/PPT reports
- `GetAnalysisStatus` - Check analysis progress
