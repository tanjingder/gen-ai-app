# Whisper Models Directory

This directory should contain the Whisper model file(s) for audio transcription.

## Required Model

**File:** `ggml-base.bin` (141 MB)

### Download Instructions

1. **Visit HuggingFace:**  
   https://huggingface.co/ggerganov/whisper.cpp/tree/main

2. **Download the model:**
   - Click on `ggml-base.bin`
   - Click the download button (↓) on the right side

3. **Place the file here:**
   ```
   mcp-servers/transcription-agent/models/ggml-base.bin
   ```

4. **Verify:**
   ```powershell
   # Windows
   dir ggml-base.bin
   
   # Linux/Mac
   ls -lh ggml-base.bin
   ```

## Why isn't this file in the repository?

The model file is **141 MB**, which exceeds GitHub's 100 MB file size limit. Therefore, it must be downloaded separately.

## Alternative Models

You can use different Whisper models depending on your needs:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `ggml-tiny.bin` | 75 MB | ⚡⚡⚡ Fast | ⭐⭐ Basic |
| **`ggml-base.bin`** | 142 MB | ⚡⚡ Medium | ⭐⭐⭐ Good (Recommended) |
| `ggml-small.bin` | 466 MB | ⚡ Slower | ⭐⭐⭐⭐ Better |
| `ggml-medium.bin` | 1.5 GB | 🐌 Slow | ⭐⭐⭐⭐⭐ Excellent |
| `ggml-large.bin` | 2.9 GB | 🐌🐌 Very Slow | ⭐⭐⭐⭐⭐⭐ Best |

All models can be downloaded from:  
https://huggingface.co/ggerganov/whisper.cpp/tree/main

## Troubleshooting

**Error: "Model not found"**
- Make sure the file is named exactly `ggml-base.bin` (or update `server.py` with your model name)
- Check the file is in the correct directory
- Verify the file downloaded completely (should be ~141 MB)

**Error: "Failed to load model"**
- The model file may be corrupted
- Re-download the model
- Ensure you have enough RAM (base model needs ~500MB RAM)

## More Information

- Whisper.cpp GitHub: https://github.com/ggerganov/whisper.cpp
- Whisper by OpenAI: https://github.com/openai/whisper
- Model details: https://huggingface.co/ggerganov/whisper.cpp
