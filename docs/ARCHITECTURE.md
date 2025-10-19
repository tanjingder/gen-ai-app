# Video Analysis AI - Software Architecture# Architecture Deep Dive



> **Last Updated:** October 19, 2025## System Overview



## 📋 Table of ContentsThe Video Analysis AI is a fully local, offline-capable desktop application that uses a multi-agent architecture coordinated through the Model Context Protocol (MCP).

- [High-Level Architecture](#high-level-architecture)

- [Query Processing Flow](#query-processing-flow)## Components

- [Component Details](#component-details)

- [Communication Protocols](#communication-protocols)### 1. Frontend (React + Tauri)

- [Data Structures](#data-structures)

- [Cache Strategy](#cache-strategy)**Technology**: React 18, TypeScript, Tailwind CSS, Tauri 1.5



---**Responsibilities**:

- User interface and interaction

## High-Level Architecture- Video file upload

- Chat-based query interface

```- Display analysis results

┌─────────────────────────────────────────┐- Download generated reports

│   Frontend (Tauri Desktop App)          │

│   React + TypeScript                    │**Key Files**:

│   ├─ Chat Interface                     │- `App.tsx` - Main application component

│   ├─ Video Upload                       │- `ChatInterface.tsx` - Chat UI with streaming support

│   └─ Report Viewer                      │- `VideoUpload.tsx` - Drag-and-drop video upload

└────────────┬────────────────────────────┘- `grpcClient.ts` - gRPC communication layer

             │ gRPC (Binary Protocol)

             │ Port 50051**Communication**: 

             ▼- Uses gRPC to communicate with backend

┌─────────────────────────────────────────┐- Tauri provides native desktop capabilities

│   Backend (Python gRPC Server)          │- Streaming responses for real-time feedback

│   ├─ Session Manager (File-based)       │

│   ├─ Video Store (In-memory + Cache)    │### 2. Backend (Python)

│   ├─ MCP Orchestrator (Intent-driven)   │

│   └─ gRPC Service Layer                 │**Technology**: Python 3.11+, FastAPI, gRPC, Ollama Client

└────────────┬────────────────────────────┘

             │**Responsibilities**:

      ┌──────┴──────┐- gRPC server hosting

      │             │- Request orchestration

      ▼             ▼- MCP client management

┌───────────┐  ┌──────────────────┐- Video file storage and metadata

│  Ollama   │  │  MCP Agents      │- Ollama LLM integration for planning

│  (Local)  │  │  (Stdio Spawn)   │

│           │  │                  │**Key Modules**:

│ llama3.2  │  │ ├─ Transcription │

│ llava     │  │ ├─ Vision        │#### `grpc_server/`

└───────────┘  │ └─ Report        │- `server.py` - gRPC server initialization

               └──────────────────┘- `service.py` - Service method implementations

```

#### `mcp/`

---- `client.py` - MCP client for calling agent tools

- `orchestrator.py` - Agent coordination logic

## Query Processing Flow

#### `models/`

### Intent-Driven Cache-First Architecture- `video_store.py` - In-memory video metadata store



```#### `utils/`

┌──────────────────┐- `config.py` - Configuration management

│   User Query     │- `setup.py` - Dependency checks and initialization

└────────┬─────────┘

         │**Communication**:

         ▼- Exposes gRPC API to frontend

┌─────────────────────────────────────┐- Makes HTTP calls to MCP servers

│ [1] ANALYZE INTENT                  │- Uses Ollama API for LLM inference

│  - Classify query type              │

│  - Determine required data          │### 3. Ollama (LLM Layer)

│  - Identify output format           │

└────────┬────────────────────────────┘**Technology**: Ollama, LLaMA 3.2, LLaVA

         │

         ▼**Responsibilities**:

┌─────────────────────────────────────┐- Query understanding and planning

│ [2] CHECK CACHE                     │- Agent selection and coordination

│  - Session folder lookup            │- Response synthesis

│  - data/sessions/{id}/cache/*.json  │- Vision analysis (via LLaVA)

└────────┬────────────────────────────┘

         │**Models Used**:

         ▼- `llama3.2` - Main reasoning and planning model

    ┌────────┐- `llava` - Vision analysis model

    │Missing?│

    └───┬────┘**Flow**:

        │1. User query → Ollama analyzes intent

   ┌────┴────┐2. Ollama creates execution plan (JSON)

   │         │3. Plan specifies which agents to invoke

  Yes       No4. Results → Ollama synthesizes final answer

   │         │

   ▼         ▼### 4. MCP Servers (Agents)

┌──────────────────┐    ┌──────────────────┐

│ [3] PLAN TOOLS   │    │ [5] SYNTHESIZE   │#### Transcription Agent

│  - Which agents? │    │  - Use cached    │

│  - Which tools?  │    │    data          │**Technology**: Python, MCP SDK, FFmpeg, (Whisper.cpp)

│  - Execution     │    │  - Generate      │

│    order         │    │    response      │**Tools**:

└───┬──────────────┘    └───┬──────────────┘- `extract_audio` - Extract audio from video using FFmpeg

    │                       │- `transcribe_audio` - Convert speech to text

    ▼                       │- `get_timestamps` - Get timestamped segments

┌──────────────────┐        │

│ [4] EXECUTE      │        │**Integration Points**:

│  - Spawn agents  │        │- FFmpeg for audio extraction

│  - Run tools     │        │- Placeholder for Whisper.cpp integration

│  - Cache results │        │- Returns structured transcription data

└───┬──────────────┘        │

    │                       │#### Vision Agent

    └───────────────────────┘

                │**Technology**: Python, MCP SDK, OpenCV, (Tesseract, LLaVA)

                ▼

        ┌───────────────┐**Tools**:

        │ [6] OUTPUT &  │- `extract_frames` - Extract key frames at intervals

        │     CACHE     │- `analyze_frame` - Describe frame content

        │  - Stream to  │- `detect_objects` - Object detection (placeholder)

        │    frontend   │- `extract_text` - OCR text extraction

        │  - Save cache │- `analyze_chart` - Chart and graph analysis (placeholder)

        └───────────────┘

```**Integration Points**:

- OpenCV for frame extraction

### Detailed Flow Example- Tesseract for OCR

- Placeholder for vision models

**User:** "Summarize this video"

#### Report Agent

```

1. ANALYZE INTENT**Technology**: Python, MCP SDK, ReportLab, python-pptx

   ├─ Intent: "summarize"

   ├─ Output: "pdf"**Tools**:

   └─ Required Data: - `create_pdf_report` - Generate PDF document

       ├─ transcription- `create_ppt_report` - Generate PowerPoint presentation

       ├─ frames- `format_content` - Structure analysis results

       ├─ visual_analysis

       └─ summary**Integration Points**:

- ReportLab for PDF generation

2. CHECK CACHE- python-pptx for PowerPoint creation

   ├─ Check: data/sessions/{id}/cache/transcription.json- Template-based formatting

   ├─ Check: data/sessions/{id}/cache/frames.json

   ├─ Check: data/sessions/{id}/cache/visual_analysis.json## Data Flow

   └─ Check: data/sessions/{id}/cache/summary.json

   ### Video Upload Flow

   Result: ❌ transcription missing

           ❌ visual_analysis missing```

           ✅ frames cached1. User drags video file

   ↓

3. PLAN TOOLS2. Frontend reads file as chunks

   ├─ Agent: transcription   ↓

   │   ├─ extract_audio3. gRPC stream to backend

   │   └─ transcribe_audio   ↓

   ├─ Agent: vision4. Backend saves to uploads/

   │   └─ analyze_frame (use cached frames)   ↓

   └─ Skip: frames (already cached)5. Extract metadata (OpenCV)

   ↓

4. EXECUTE AGENTS6. Store in video_store

   ├─ Spawn: transcription-agent via stdio   ↓

   │   └─ Cache: transcription.json7. Return video_id to frontend

   ├─ Spawn: vision-agent via stdio```

   │   └─ Cache: visual_analysis.json

   └─ All results collected### Query Flow



5. SYNTHESIZE```

   ├─ Aggregate all data1. User types question

   ├─ Generate structured summary with Ollama   ↓

   └─ Format for report2. Frontend sends ChatRequest

   ↓

6. OUTPUT & CACHE3. Backend receives query

   ├─ Generate PDF report   ↓

   ├─ Cache: summary.json4. Ollama plans execution

   └─ Return: PDF path to frontend   ↓

```5. Backend calls MCP agents

   ↓

---6. Agents execute tools

   ↓

## Component Details7. Results collected

   ↓

### 1. Frontend (Tauri + React)8. Ollama synthesizes answer

   ↓

**Tech Stack:**9. Stream response to frontend

- **Framework:** React 18 + TypeScript```

- **Desktop:** Tauri 1.5 (Rust)

- **Styling:** Tailwind CSS### Agent Execution Flow

- **Communication:** gRPC (tonic + prost)

```

**Key Responsibilities:**1. Orchestrator receives plan

- User interface and interactions   ↓

- Video file upload (chunked streaming)2. For each agent in plan:

- Real-time chat with AI   │

- Report viewing and download   ├─> MCP Client calls agent tool

- Session management   │   ↓

   │   Agent processes request

**File Structure:**   │   ↓

```   │   Agent returns result

frontend/   │   ↓

├── src/   └─> Result added to collection

│   ├── App.tsx                 # Main component   ↓

│   ├── components/3. All results aggregated

│   │   ├── ChatInterface.tsx   # Chat UI   ↓

│   │   └── VideoUpload.tsx     # Upload UI4. Return to orchestrator

│   └── services/```

│       └── grpc.ts             # gRPC client

└── src-tauri/## Communication Protocols

    ├── src/

    │   └── main.rs             # Rust gRPC client### gRPC (Frontend ↔ Backend)

    └── Cargo.toml              # Dependencies

```**Protocol**: HTTP/2-based RPC



**gRPC Client (Rust):****Messages**:

```rust- `ChatRequest` / `ChatResponse` - Chat messages

// Native gRPC client using tonic- `VideoChunk` / `VideoUploadResponse` - File upload

pub async fn chat(request: ChatRequest) -> Result<ChatResponse> {- `VideoQuery` / `QueryResponse` - Video queries

    let mut client = VideoAnalysisClient::connect("http://localhost:50051").await?;- `ReportRequest` / `ReportResponse` - Report generation

    let response = client.chat(request).await?;- `StatusRequest` / `AnalysisStatus` - Progress tracking

    Ok(response.into_inner())

}**Features**:

```- Bidirectional streaming

- Protocol buffer serialization

---- Type safety

- High performance

### 2. Backend (Python gRPC Server)

### MCP (Backend ↔ Agents)

**Tech Stack:**

- **Language:** Python 3.11+**Protocol**: Model Context Protocol over HTTP

- **RPC:** gRPC (grpcio)

- **LLM:** Ollama API client**Structure**:

- **Agents:** MCP SDK 0.9.1 (stdio)```json

{

**Key Responsibilities:**  "method": "tools/call",

- gRPC API endpoints  "params": {

- Session and file management    "name": "tool_name",

- MCP orchestration (intent-driven)    "arguments": { ... }

- Video metadata extraction  }

- Cache management}

```

**File Structure:**

```**Tools Discovery**:

backend/```json

├── main.py                     # Entry point{

├── src/  "method": "tools/list",

│   ├── grpc_server/  "params": {}

│   │   ├── server.py           # gRPC server setup}

│   │   ├── service.py          # Service methods```

│   │   └── video_analysis_pb2.py  # Generated proto

│   ├── mcp/**Features**:

│   │   ├── client.py           # MCP agent client- Standardized agent communication

│   │   └── orchestrator.py     # Intent-driven orchestrator- Tool discovery

│   ├── models/- JSON-based messages

│   │   └── video_store.py      # Video metadata store- Extensible

│   └── utils/

│       ├── config.py           # Configuration### Ollama API (Backend ↔ LLM)

│       └── session_manager.py  # Session handling

└── requirements.txt**Protocol**: HTTP REST

```

**Endpoints**:

**Session Structure:**- `POST /api/chat` - Chat completion

```- `POST /api/generate` - Text generation

data/sessions/{session_id}/- `GET /api/tags` - List models

├── session.json            # Session metadata

├── messages.json           # Chat history**Features**:

├── uploads/                # Video files- Local inference

│   └── {video_id}_{filename}- No external API calls

├── cache/                  # Analysis cache- Model management

│   ├── transcription.json- Streaming responses

│   ├── frames.json

│   ├── visual_analysis.json## Security Considerations

│   └── summary.json

├── temp/                   # Temporary files### Local-First Architecture

│   └── {video_id}_audio.wav

└── reports/                # Generated reports- All data stays on device

    └── {title}.pdf- No cloud API calls

```- No telemetry or tracking

- User data never leaves machine

---

### File Handling

### 3. MCP Orchestrator (Intent-Driven)

- Videos stored in configured directory

**Core Logic:**- Temporary files cleaned up

- File size limits enforced

```python- Path traversal protection

class MCPOrchestratorV2:

    async def process_query(self, query: str, video_id: str, session: Session):### Desktop Security

        # Step 1: Analyze intent

        intent = await self.analyze_intent(query)- Tauri provides sandboxing

        # Returns: {- No web-based vulnerabilities

        #   "intent": "summarize",- Native OS security features

        #   "output_type": "pdf", - Code signing in production

        #   "required_data": ["transcription", "visual_analysis"]

        # }## Scalability

        

        # Step 2: Check cache### Current Limitations

        cache_status = self.check_cache(intent["required_data"], session.cache)

        # Returns: {- Single-threaded video processing

        #   "transcription": False,  # Missing- In-memory video store

        #   "visual_analysis": False # Missing- Sequential agent execution

        # }- Limited by local hardware

        

        # Step 3: Plan tools### Future Optimizations

        tools_plan = self.plan_missing_tools(intent["required_data"], cache_status)

        # Returns: [- Parallel agent execution

        #   {"agent": "transcription", "tool": "extract_audio"},- Persistent video database

        #   {"agent": "transcription", "tool": "transcribe_audio"},- GPU acceleration

        #   {"agent": "vision", "tool": "analyze_frame"}- Batch processing

        # ]- Result caching

        

        # Step 4: Execute tools## Extension Points

        results = await self.execute_tools(tools_plan, video_id, session.cache)

        # Spawns agents, runs tools, caches results### Adding New Agents

        

        # Step 5: Synthesize1. Create new MCP server

        response = await self.synthesize_response(intent, results)2. Define tools in MCP format

        3. Register in orchestrator

        # Step 6: Cache and return4. Update system prompt

        session.cache.set("summary", response)5. Configure in `.env`

        return response

```### Adding New LLM Models



**Intent Analysis (Ollama):**1. Pull model via Ollama

```python2. Update configuration

prompt = f"""Analyze this user query: "{query}"3. Adjust prompts if needed

4. Test compatibility

Return JSON:

{{### Custom Report Templates

  "intent": "summarize | transcribe | visual_analysis | chat",

  "output_type": "text | pdf | ppt",1. Extend report agent

  "required_data": ["transcription", "frames", "visual_analysis"]2. Add template system

}}3. Expose via tool parameters

"""4. Update frontend UI

response = ollama.chat(model="llama3.2", messages=[...])

```## Performance Characteristics



---### Video Upload

- Speed: ~10-50 MB/s

### 4. MCP Agents (Stdio Spawn)- Limited by: Disk I/O



**Architecture:** Agents spawn on-demand via stdio (not HTTP servers)### Transcription

- Speed: ~1-5x realtime

#### Transcription Agent- Limited by: Model size, CPU/GPU

```

Tools:### Frame Analysis

├─ extract_audio(video_path, output_path)- Speed: ~0.5-2 frames/sec

│   └─ Uses: FFmpeg- Limited by: Vision model, GPU

├─ transcribe_audio(audio_path)

│   └─ Uses: Whisper (future)### Report Generation

└─ Returns: {text, timestamps}- Speed: <1 second

```- Limited by: Content complexity



#### Vision Agent## Monitoring and Debugging

```

Tools:### Logs

├─ extract_frames(video_path, interval, output_dir)

│   └─ Uses: OpenCV- Backend: `backend/app.log`

├─ analyze_frame(frame_path)- MCP Servers: stdout/stderr

│   └─ Uses: Ollama LLaVA- Frontend: Developer console

├─ detect_objects(frame_path)

│   └─ Uses: YOLO (future)### Health Checks

└─ extract_text(frame_path)

    └─ Uses: Tesseract OCR- Ollama connection check on startup

```- MCP server availability check

- gRPC connection status

#### Report Agent

```### Error Handling

Tools:

├─ create_pdf_report(content, output_path)- Graceful degradation

│   └─ Uses: ReportLab- User-friendly error messages

├─ create_ppt_report(content, output_path)- Detailed logging for debugging

│   └─ Uses: python-pptx- Retry logic for transient failures

└─ Returns: {path, size_bytes}
```

**Spawn Flow:**
```python
# Backend spawns agent on-demand
process = subprocess.Popen(
    ["python", "mcp-servers/vision-agent/server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Send MCP request via stdin
request = {"method": "tools/call", "params": {...}}
process.stdin.write(json.dumps(request).encode())

# Read response from stdout
response = json.loads(process.stdout.readline())
```

---

## Communication Protocols

### gRPC (Frontend ↔ Backend)

**Protocol:** HTTP/2 Binary RPC  
**Port:** 50051  
**Format:** Protocol Buffers

**Key Messages:**
```protobuf
// Chat
message ChatRequest {
    string user_message = 1;
    string session_id = 2;
    optional string video_id = 3;
}

message ChatResponse {
    string ai_response = 1;
    bool success = 2;
    optional string error = 3;
}

// Video Upload
message VideoChunk {
    bytes chunk = 1;
    string filename = 2;
    int32 chunk_index = 3;
    optional VideoMetadata metadata = 4;
}

// Session
message SessionResponse {
    string session_id = 1;
    optional string video_id = 6;
    optional VideoMetadata video_metadata = 7;
}
```

**Advantages:**
- High performance (binary serialization)
- Type safety (schema validation)
- Bidirectional streaming
- Language agnostic

---

### MCP (Backend ↔ Agents)

**Protocol:** Model Context Protocol (stdio)  
**Format:** JSON-RPC over stdin/stdout

**Tool Call:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "extract_frames",
    "arguments": {
      "video_path": "data/sessions/{id}/uploads/video.mp4",
      "interval_seconds": 30,
      "output_dir": "data/sessions/{id}/temp"
    }
  }
}
```

**Tool Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "frames": [
      {"frame_index": 0, "timestamp": 0.0, "path": "...frame_0.jpg"},
      {"frame_index": 1, "timestamp": 30.0, "path": "...frame_1.jpg"}
    ]
  }
}
```

**Advantages:**
- Standardized agent communication
- Process isolation (agents crash independently)
- Easy debugging (JSON messages)
- Extensible (add new tools easily)

---

### Ollama API (Backend ↔ LLM)

**Protocol:** HTTP REST  
**Port:** 11434  
**Format:** JSON

**Chat Completion:**
```python
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'system', 'content': 'You are a video analysis assistant.'},
        {'role': 'user', 'content': 'Summarize this video based on: ...'}
    ]
)
```

**Vision Analysis:**
```python
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe this image concisely',
        'images': [base64_encoded_image]
    }]
)
```

---

## Data Structures

### Video Metadata
```python
@dataclass
class VideoInfo:
    id: str
    filename: str
    file_path: str
    duration: float          # seconds
    resolution: str          # "1920x1080"
    fps: float
    file_size: int           # bytes
    upload_time: datetime
```

### Session Data
```json
{
  "session_id": "uuid",
  "created_at": "2025-10-19T10:00:00Z",
  "last_active": "2025-10-19T10:30:00Z",
  "video_id": "uuid",
  "video_filename": "example.mp4",
  "video_metadata": {
    "duration_ms": 150000,
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "file_size": 52428800
  },
  "message_count": 15,
  "analysis_complete": true
}
```

### Cache Entry
```json
{
  "transcription": {
    "text": "Full transcription text...",
    "timestamps": [...],
    "generated_at": "2025-10-19T10:15:00Z"
  },
  "visual_analysis": {
    "frame_analyses": [
      {
        "frame": 0,
        "timestamp": 0.0,
        "description": "A person coding on a laptop",
        "objects": ["person", "laptop", "desk"],
        "text": "console.log('Hello');"
      }
    ],
    "generated_at": "2025-10-19T10:20:00Z"
  }
}
```

---

## Cache Strategy

### Cache-First Philosophy

**Benefits:**
1. **Speed:** Instant responses for repeated queries
2. **Cost:** No redundant processing
3. **Offline:** Works without internet after initial analysis
4. **Consistency:** Same data across requests

**Cache Locations:**
```
data/sessions/{session_id}/cache/
├── transcription.json      # Audio-to-text
├── frames.json             # Extracted frames metadata
├── visual_analysis.json    # Frame descriptions
├── objects.json            # Detected objects
├── text.json               # OCR results
├── summary.json            # Generated summaries
├── pdf_report.json         # PDF metadata
└── ppt_report.json         # PPT metadata
```

**Cache Invalidation:**
- Automatic: When video changes
- Manual: User can clear cache
- TTL: Optional time-based expiry (future)

**Cache Hit Example:**
```
User: "What's in this video?"
├─ Check cache: summary.json ✅ EXISTS
└─ Return: Cached summary (< 10ms)

User: "Create a PDF report"
├─ Check cache: summary.json ✅ EXISTS
├─ Check cache: transcription.json ✅ EXISTS
├─ Check cache: visual_analysis.json ✅ EXISTS
└─ Generate: PDF using cached data (< 1s)
```

---

## Error Handling

### Graceful Degradation

```python
try:
    # Try primary method
    result = await agent.execute_tool(...)
except AgentTimeoutError:
    # Fallback to cache or default
    result = cache.get("fallback") or default_result
except AgentCrashError:
    # Log and notify user
    logger.error("Agent crashed")
    return {"error": "Analysis temporarily unavailable"}
```

### User-Facing Errors

**Types:**
- `VIDEO_NOT_FOUND` - Video file missing
- `AGENT_TIMEOUT` - Agent took too long
- `ANALYSIS_FAILED` - Processing error
- `CACHE_CORRUPT` - Cache data invalid

**Example:**
```typescript
if (response.error) {
  toast.error(`Analysis failed: ${response.error}`);
  // Show retry button
}
```

---

## Performance Characteristics

| Operation | Speed | Bottleneck |
|-----------|-------|------------|
| Video Upload | 10-50 MB/s | Disk I/O |
| Chat Response | 50-200 ms | Ollama inference |
| Frame Extraction | 5-20 fps | CPU/Disk |
| Transcription | 1-5x realtime | Whisper model |
| Vision Analysis | 0.5-2 fps | LLaVA model |
| PDF Generation | < 1s | Content size |
| Cache Hit | < 10ms | Disk read |

---

## Security & Privacy

### Local-First Architecture
- ✅ All processing on-device
- ✅ No cloud API calls
- ✅ No telemetry
- ✅ User data never leaves machine

### File Security
- Videos stored in configured directory
- Temporary files cleaned after processing
- File size limits enforced
- Path traversal protection

### Desktop Sandboxing
- Tauri provides OS-level sandboxing
- Limited file system access
- No arbitrary code execution
- Signed binaries in production

---

## Future Enhancements

### Performance
- [ ] Parallel agent execution
- [ ] GPU acceleration for vision
- [ ] Persistent video database (SQLite)
- [ ] Smart cache preloading

### Features
- [ ] Multi-video comparison
- [ ] Custom report templates
- [ ] Export to more formats (JSON, CSV)
- [ ] Video timeline scrubbing

### Architecture
- [ ] Plugin system for custom agents
- [ ] Distributed processing (multiple machines)
- [ ] Real-time collaboration
- [ ] Cloud sync (optional)

---

**For setup instructions, see:** [QUICKSTART.md](../QUICKSTART.md)  
**For user guide, see:** [README.md](../README.md)
