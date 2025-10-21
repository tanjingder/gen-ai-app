# Local AI Video Analysis Desktop Application

A fully offline AI desktop application for analyzing and querying short video files with multi-agent architecture.

## 🏗️ Architecture

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

### Core Requirements (Backend & Frontend)

Before starting, install these required software:

| Software | Version | Download Link | Notes |
|----------|---------|---------------|-------|
| **Git** | Latest | [Download](https://git-scm.com/downloads/win) | Version control |
| **Ollama** | Latest | [Download](https://ollama.com/download/windows) | Local AI runtime |
| **Python** | 3.11+ | [Microsoft Store](https://apps.microsoft.com/store/detail/python-311/9NRWMJP3717K) | Backend runtime |
| **Node.js** | 18+ | [Download](https://nodejs.org/en/download) | Frontend tools |
| **Visual Studio** | 2022+ | [Download](https://visualstudio.microsoft.com/) | C++ build tools (Desktop development with C++) |
| **Rust** | Latest | [Download](https://rustup.rs/) | Tauri framework |

### Agent Tools Requirements

Additional tools required for MCP agents:

| Tool | Purpose | Download Link | Installation Notes |
|------|---------|---------------|-------------------|
| **FFmpeg** | Audio extraction | [Download](https://www.gyan.dev/ffmpeg/builds/) | Extract `ffmpeg-release-essentials.zip` and add `bin` folder to System Path |
| **Tesseract OCR** | Text extraction | [Download](https://sourceforge.net/projects/tesseract-ocr.mirror/files/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe/download) | Add installation folder to System Path |
| **Protoc** | Protocol buffers | [Download](https://github.com/protocolbuffers/protobuf/releases) | Extract and add `bin` folder to System Path |
| **whisper.cpp** | Audio transcription | [GitHub Releases](https://github.com/ggerganov/whisper.cpp/releases) | Download `whisper-bin-x64.zip`, extract to `mcp-servers/transcription-agent/whisper.cpp/` |
| **ggml-base.bin** | Whisper model | [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/tree/main) | Download `ggml-base.bin` (~142 MB) to `mcp-servers/transcription-agent/models/` |

### Ollama Models

After installing Ollama, download required AI models:

```powershell
ollama serve
ollama pull llama3.2    # Main reasoning model (~4.7 GB)
ollama pull llava       # Vision analysis model (~4.5 GB)
```

### System Requirements

- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 20GB free space (for models and cache)
- **OS:** Windows 10/11 (64-bit)

> **💡 Tip:** Run `verify-setup.bat` after installation to check if all prerequisites are properly installed.

## Quick Start (5 Minutes)

### 1. Clone Repository

```powershell
git clone https://github.com/tanjingder/gen-ai-app.git
cd gen-ai-app
```

### 2. Verify Prerequisites

Check if all required tools are installed:

```batch
scripts\verify-setup.bat
```

This will check:
- Python, Node.js, Rust versions
- Git, Ollama availability
- FFmpeg, Tesseract, Protoc in PATH
- Visual Studio C++ tools

### 3. Run Setup

```batch
setup.bat
```

This will:
1. Set up Python backend with shared virtual environment
2. Install all MCP agent dependencies (using shared venv)
3. Set up Tauri frontend (Rust + Node.js dependencies)
4. Compile protocol buffers

**Duration:** 5-10 minutes (depending on internet speed)

### 4. Start Application

```batch
start.bat
```

This will:
1. Start Ollama (if not already running)
2. Start backend (gRPC server on port 50051)
3. Start frontend (Tauri desktop application)

**Note:** MCP agents spawn automatically on-demand. No separate server processes needed!

---

## 🔧 Manual Setup (If Automated Setup Fails)

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

## Project Reflection

### ✅ What Works?

**1. Multi-Format Report Generation**
- Successfully generates structured PDF and PowerPoint reports from video analysis
- Includes comprehensive sections: transcript, visual analysis, object detection, and extracted text
- Automated formatting with proper styling and organization

**2. Audio Transcription**
- Accurate speech-to-text conversion using Whisper.cpp (ggml-base.bin model)
- Extracts full audio track from video files
- Handles multiple audio formats through FFmpeg preprocessing

**3. Visual Content Analysis**
- **Object Detection**: YOLOv8 successfully detects objects across 10 sampled frames
  - Provides confidence scores and instance counts
  - LLM generates natural language summaries of detected objects
- **Frame Description**: LLaVA vision model analyzes visual content
  - Describes scenes, identifies elements (graphs, charts, code, presentations)
  - Provides detailed frame-by-frame breakdowns with timestamps
- **OCR Text Extraction**: Tesseract extracts on-screen text from 10 frames
  - Cleans and filters OCR noise
  - LLM analyzes extracted text for key information

**4. Intelligent Query Processing**
- Intent classification system routes queries to appropriate tools
- Cache-first architecture prevents redundant processing
- Supports multiple query types: summarization, object detection, visual analysis, text extraction, and general questions

**5. Session Management**
- Persistent chat sessions with automatic video-session binding
- Efficient caching of analysis results (frames, transcript, objects, text)
- Quick retrieval for follow-up questions without re-processing

---

### ❌ What Doesn't Work?

**1. Inconsistent LLM Responses**
- **Issue**: Sometimes the model replies "nothing were observed" even when data exists
- **Cause**: LLM may not properly interpret cached data or receives incomplete context
- **Examples**: 
  - Visual analysis returns "No visual analysis available" when frames exist but weren't analyzed
  - Object detection shows contradictory answers (e.g., "No charts" but mentions "bar graphs" in summary)

**2. Intent Classification Failures**
- **Issue**: Similar queries get classified differently
- **Examples**:
  - "Are there any graphs?" → `visual_analysis` ✅
  - "Are there any graphs or charts?" → `chat` ❌ (should be `visual_analysis`)
- **Impact**: Wrong intent = wrong tools executed = poor responses

**3. LLM Tool Usage Confusion**
- **Issue**: LLM doesn't always understand when/how to use available tools
- **Examples**:
  - User asks "What's in the video?" → LLM tries to answer without analyzing frames
  - Should execute `analyze_frame` tool but generates generic response instead
- **Cause**: Insufficient tool descriptions in prompts and unclear execution flow

---

### 🚀 Potential Improvements

**1. More Flexible, Natural LLM Responses**

**Current State**: Rigid, structured output
    🎬 Visual Analysis
    Analyzed 6 frames

    Direct Answer: [Yes/No answer]
    Brief Summary: [2-3 sentences]
    Key Visual Elements:

    Item 1
    Item 2

**Improved State**: Conversational, context-aware responses
    Based on analyzing 6 frames throughout your video, I can see several bar graphs
    appearing around the 1:16 mark showing price trends, and another at 1:54
    displaying religious demographics over time. The video appears to be an
    educational presentation about data visualization techniques.

    The most prominent charts are:

    A price trend graph with daily data (frame 3)
    A demographics bar chart showing religious affiliations by age (frame 4)
    A 5K race runners distribution chart (frame 5)
    Would you like me to provide more details about any specific chart?

**Benefits**:
- ✅ More engaging and human-like
- ✅ Adapts tone to user's question style
- ✅ Offers follow-up suggestions
- ✅ Less repetitive formatting

**2. Dynamic Response Length**
- Short answers for simple yes/no questions
- Detailed analysis when user asks for in-depth information
- Progressive disclosure (summary first, expand on request)

**3. Multi-Modal Synthesis**
- Combine transcript + visual + objects + text into cohesive narrative
- Example: "At 1:30, the speaker explains event loops while showing a diagram with multiple processes (detected: 3 rectangles, 2 arrows)"

**4. Interactive Clarifications**
- When LLM is uncertain, ask clarifying questions
- Example: "I see some graphical elements at 1:30. Are you asking about the code diagram or the data visualization chart?"

---

### 💪 Encountered Challenges

#### **1. gRPC Connection Stability**

**Challenge**: No prior experience building gRPC-based MCP architecture
- Struggled with proper server initialization and lifecycle management
- Unclear how to handle connection timeouts and retries
- Difficulty debugging connection failures (no clear error messages)

**Learning Curve**:
- Understanding Protocol Buffers schema definitions
- Managing bidirectional streaming vs unary calls
- Implementing proper error handling and graceful shutdowns

#### **2. LLM Response Accuracy**

**Challenge**: Local LLMs (Ollama llama3.2, llava) produce inconsistent results

**Specific Issues**:

a) **Hallucinations**
- LLM invents details not present in video
- Example: Claims to see "presenter speaking" when only slides are shown
- Vision model sometimes misidentifies objects (e.g., "laptop" when it's a monitor)

b) **Context Window Limitations**
- Long transcripts get truncated, losing important details
- Ollama llama3.2 has ~8K token context limit
- Can't include full transcript + all frame descriptions + user query in one prompt

c) **Inconsistent Formatting**
- Sometimes follows structured prompt format perfectly
- Other times ignores instructions and generates freeform text
- JSON parsing failures when LLM adds extra commentary

d) **Contradictory Logic**
- Says "No charts" in Direct Answer but lists "bar graphs" in Summary
- Claims "nothing detected" despite object detection results showing 52 objects

#### **3. Tool Orchestration Complexity**

**Challenge**: LLM doesn't inherently know when/how to use tools

**Initial Approach**: Let LLM decide which tools to call
- **Problem**: LLM often skips tool usage and tries to answer directly
- **Example**: User asks "What objects are in the video?" → LLM responds "I cannot see the video" instead of calling `detect_objects`

**Current Solution**: Intent classification + forced execution
- Analyze user query intent first
- Map intent to required tools/data
- Execute tools regardless of LLM's preference
- Pass results to LLM for synthesis

**Why This Was Hard**:
- Designing robust intent classification (many edge cases)
- Creating comprehensive examples for each intent type
- Balancing flexibility vs rigid tool execution

### ⏰ What Could Be Achieved with More Time

#### **1. Enhanced Response Quality**

**A. Better LLM Integration**
- **Fine-tune local model** on video analysis tasks
  - Train on examples of good vs poor responses
  - Teach model to cite specific frames/timestamps
  - Improve object recognition vocabulary
- **Implement RAG (Retrieval-Augmented Generation)**
  - Store analysis results in vector database
  - Retrieve relevant context for each query
  - Reduce hallucinations by grounding responses in actual data
- **Multi-step reasoning**
  - Break complex queries into sub-questions
  - Example: "What's the main topic?" → Analyze transcript + visual + objects → Synthesize
  - Chain-of-thought prompting for better logic

**B. Smarter Context Management**
- **Sliding window for long videos**
  - For >10min videos, analyze in segments
  - Summarize each segment, then create overall summary
- **Hierarchical caching**
  - Frame-level cache (raw analysis)
  - Segment-level cache (summarized chunks)
  - Video-level cache (overall summary)
- **Dynamic context selection**
  - Only include relevant cached data in LLM prompt
  - Example: If user asks about timestamp 2:30, only load transcript/frames near that time

#### **2. Model & Tool Improvements**

**A. Better Vision Models**
 **Upgrade to LLaVA 1.6 or GPT-4V** (if available locally)
- Higher accuracy for chart/graph recognition
 **Specialized models for specific tasks**
- Chart understanding model (e.g., ChartQA)
- Presentation slide analyzer

**B. Enhanced Object Detection**
 **Custom YOLOv8 training**
- Fine-tune on domain-specific objects (charts, code, diagrams)
- Improve confidence scores for ambiguous objects
 **Object tracking across frames**
- Identify when the same object appears in multiple frames
- "Person in frames 1-4 appears to be the same individual"
 **Scene classification**
- Detect scene types: presentation, coding demo, outdoor, interview
- Adjust analysis strategy based on scene type

**C. Improved Transcript Analysis**
 **Speaker diarization** (who said what)
- Identify multiple speakers
- Attribute quotes correctly
 **Sentiment analysis**
- Detect tone: educational, promotional, informative
 **Key phrase extraction**
- Automatically highlight important terms
- Generate glossary of technical terms mentioned

---

### 📊 Summary

**Current State**: Functional MVP with core features working but inconsistent reliability

**Main Achievements**:
- ✅ Multi-modal analysis (audio, visual, text, objects)
- ✅ Session-based caching for efficient re-querying
- ✅ Report generation in multiple formats

**Key Limitations**:
- ❌ LLM response inconsistency and hallucinations
- ❌ Rigid response formatting
- ❌ No real-time or progressive analysis

**Biggest Learning**:
Building an AI orchestration system is more about **reliable infrastructure** and **careful prompt engineering** than raw model capabilities. The best LLM in the world won't help if the tool execution is unreliable or the prompts are ambiguous.