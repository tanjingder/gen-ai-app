# Transcription Agent MCP Server

An MCP server for audio extraction and speech-to-text transcription.

## Features

- Extract audio from video files using ffmpeg
- Transcribe audio to text (placeholder for whisper.cpp integration)
- Get timestamped transcription segments

## Prerequisites

- Python 3.11+
- ffmpeg installed and in PATH
- (Optional) whisper.cpp for real transcription

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

```powershell
python server.py
```

## Integrating Whisper.cpp

To enable real transcription:

1. Clone and build whisper.cpp:
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

2. Download a model:
```bash
bash ./models/download-ggml-model.sh base.en
```

3. Update the `transcribe_audio` method in `server.py` to call whisper.cpp

## Tools

### extract_audio
Extract audio track from video file.

**Input:**
- `video_path` (string): Path to video file
- `output_path` (string): Output audio file path

### transcribe_audio
Transcribe audio to text with timestamps.

**Input:**
- `audio_path` (string): Path to audio file
- `language` (string, optional): Language code (default: "en")

### get_timestamps
Get timestamped segments from transcription.

**Input:**
- `audio_path` (string): Path to audio file
