# Video Analysis AI - Architecture Overview

> **Last Updated:** October 20, 2025

## System Architecture

```
┌─────────────────────────────────────────┐
│   Frontend (Tauri Desktop App)          │
│   React + TypeScript                    │
└────────────┬────────────────────────────┘
             │ gRPC (Port 50051)
             ▼
┌─────────────────────────────────────────┐
│   Backend (Python gRPC Server)          │
│   ├─ Session Manager                    │
│   ├─ Video Store                        │
│   └─ MCP Orchestrator                   │
└────────────┬────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌───────────┐  ┌──────────────────┐
│  Ollama   │  │  MCP Agents      │
│ llama3.2  │  │ ├─ Transcription │
│ llava     │  │ ├─ Vision        │
└───────────┘  │ └─ Report        │
               └──────────────────┘
```

## Complete Process Flow

### Query Processing Example
```
User Query: "Summarize this video"
   
[1] ANALYZE INTENT (Ollama)  Intent: "summarize"
   
[2] CHECK CACHE  Missing data
   
[3] EXECUTE AGENTS  Spawn transcription + vision agents
   
[4] SYNTHESIZE RESPONSE (Ollama)  Generate summary
   
[5] RETURN TO USER  Stream response to frontend
```

## Key Components

- **Frontend**: React + Tauri desktop app
- **Backend**: Python gRPC server + MCP orchestrator  
- **Ollama**: Local LLM (llama3.2 + llava models)
- **MCP Agents**: Transcription, Vision, Report (spawn on-demand)

## Session Structure

```
data/sessions/{session_id}/
 uploads/video.mp4
 cache/
    transcription.json
    visual_analysis.json
    summary.json
 reports/summary.pdf
```

## Communication

- **Frontend  Backend**: gRPC (binary, high performance)
- **Backend  Agents**: MCP protocol (JSON-RPC over stdio)
- **Backend  Ollama**: HTTP REST API (localhost:11434)

## Cache Strategy

```
Query  Check Cache  Hit? Return : Process  Cache  Return
```

**Benefits**: Instant responses, no redundant processing, offline capability

---

**For setup:** [QUICKSTART.md](../QUICKSTART.md) | **For usage:** [README.md](../README.md)
