# Vision Agent MCP Server

An MCP server for video frame analysis, object detection, and OCR.

## Features

- Extract frames from videos at specified intervals
- Analyze frame content (placeholder for vision models)
- Detect objects in frames (placeholder for YOLO/detection models)
- Extract text using OCR (Tesseract)
- Analyze charts and graphs (placeholder)

## Prerequisites

- Python 3.11+
- OpenCV
- (Optional) Tesseract OCR
- (Optional) Vision models like LLaVA via Ollama

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Install Tesseract (Optional)

For Windows:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH

## Running

```powershell
python server.py
```

## Integrating Vision Models

### Using LLaVA with Ollama

```bash
ollama pull llava
```

Then update the `analyze_frame` method to call Ollama's llava model.

### Using YOLO for Object Detection

1. Download YOLO weights
2. Integrate with OpenCV DNN module
3. Update `detect_objects` method

## Tools

### extract_frames
Extract key frames from video at intervals.

**Input:**
- `video_path` (string): Path to video file
- `interval_seconds` (number, optional): Seconds between frames (default: 1.0)

### analyze_frame
Analyze frame content with optional prompt.

**Input:**
- `frame_path` (string): Path to frame image
- `prompt` (string, optional): Analysis guidance

### detect_objects
Detect objects in frame.

**Input:**
- `frame_path` (string): Path to frame image

### extract_text
Extract text using OCR.

**Input:**
- `frame_path` (string): Path to frame image

### analyze_chart
Analyze charts and graphs.

**Input:**
- `frame_path` (string): Path to frame image
