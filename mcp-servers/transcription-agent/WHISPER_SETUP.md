# Whisper.cpp Setup Guide

## You Already Have
✅ Model file: `models/ggml-base.bin`

## What You Need
❌ whisper.cpp executable

## Quick Setup (Recommended)

### Step 1: Download whisper.cpp
1. Go to: https://github.com/ggerganov/whisper.cpp/releases
2. Download the Windows release (look for `whisper-bin-x64.zip` or similar)
3. Extract the ZIP file

### Step 2: Copy to Project
Copy the extracted folder contents to:
```
mcp-servers/transcription-agent/whisper.cpp/
```

Your structure should look like:
```
mcp-servers/
└── transcription-agent/
    ├── models/
    │   └── ggml-base.bin          ✅ (you have this)
    ├── whisper.cpp/
    │   ├── main.exe               ❌ (you need this)
    │   └── ...other files
    ├── server.py
    └── requirements.txt
```

### Step 3: Test Installation

**Quick Test - Whisper.cpp:**
```powershell
cd mcp-servers\transcription-agent
.\whisper.cpp\main.exe --help
```
If you see help output, whisper.cpp is installed correctly!

**Full Test - Agent Functionality:**
```powershell
cd mcp-servers\transcription-agent
.\test-agent.bat
```
This will verify:
- ✅ whisper.cpp executable exists
- ✅ Model file is present
- ✅ Agent methods work correctly
- ✅ Error handling functions properly

---

## Alternative: Build from Source

If you have Visual Studio or MinGW installed:

```powershell
cd mcp-servers\transcription-agent
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# Option A: Using CMake + Visual Studio
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Option B: Using make (MinGW)
make
```

---

## What the Code Does Now

The updated `server.py` will:
1. ✅ Check if `whisper.cpp/main.exe` exists
2. ✅ Check if your model exists at `models/ggml-base.bin`
3. ✅ Run whisper.cpp with these parameters:
   - `-m models/ggml-base.bin` - Use your model
   - `-f <audio_file>` - Input audio
   - `-l en` - Language (default English)
   - `-oj` - Output JSON for structured results
   - `-t 4` - Use 4 CPU threads
   - `-np` - No progress output (cleaner)
4. ✅ Parse the JSON output to extract:
   - Full transcription text
   - Timestamped segments (start, end, text)
5. ✅ Return results in structured format

## Error Handling

If whisper.cpp is not found, the agent will return helpful error messages with setup instructions.

## Testing

Once whisper.cpp is installed, you can test with:
```python
# In Python
result = await agent.transcribe_audio("path/to/audio.wav")
print(result)
```

The output will include:
```json
{
  "success": true,
  "text": "Full transcription text here...",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "First segment"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Second segment"
    }
  ]
}
```

## Model Information

Your `ggml-base.bin` model:
- Size: ~150MB
- Language: Multi-language (supports English, Spanish, French, etc.)
- Quality: Good balance of speed vs accuracy
- Suitable for: Most video transcription tasks

## Troubleshooting

**Error: "whisper.cpp not found"**
- Make sure `main.exe` is at `whisper.cpp/main.exe`
- Check you extracted the full whisper.cpp package

**Error: "Model not found"**
- Verify `models/ggml-base.bin` exists (you should already have this)

**Error: "Transcription timeout"**
- Audio is >5 minutes or very complex
- Try shorter segments or increase timeout in code

**Poor Quality Transcription:**
- Consider upgrading to `ggml-large.bin` for better accuracy
- Ensure audio is clear (16kHz sample rate works best)
- Check the language parameter matches your audio
