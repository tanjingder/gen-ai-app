# Scripts Directory

This directory contains all batch scripts for setting up and running the Video Analysis AI application.

## 📋 Setup Scripts

### `setup-all.bat`
Complete setup for all components. **Run this first!**
- Sets up backend Python environment
- Installs MCP server dependencies
- Sets up frontend (Node.js + Rust)
- Compiles protocol buffers

**Usage:**
```batch
.\scripts\setup-all.bat
```

### `setup-backend.bat`
Sets up only the Python backend
- Creates virtual environment
- Installs Python dependencies
- Compiles proto files
- Creates .env file

### `setup-frontend.bat`
Sets up only the Tauri frontend
- Installs npm dependencies
- Checks Rust installation

### `setup-mcp-servers.bat`
Sets up MCP agent servers
- Installs dependencies for transcription agent
- Installs dependencies for vision agent
- Installs dependencies for report agent
- Uses shared backend virtual environment

## 🚀 Start Scripts

### `start-all.bat`
Starts the complete application. **Use this to run the app!**
- Starts backend gRPC server
- Launches desktop application
- MCP agents spawn automatically as needed

**Usage:**
```batch
.\scripts\start-all.bat
```

### `start-backend.bat`
Starts only the backend server
- Runs gRPC server on port 50051
- MCP agents available for requests

### `start-mcp-servers.bat`
Information script (legacy)
- MCP servers now use stdio protocol
- No separate server processes needed
- Agents spawn on-demand

## 🔧 Utility Scripts

### `verify-setup.bat`
Verifies all prerequisites are properly installed
- Checks core requirements (Git, Python, Node.js, Rust, Ollama, VS C++)
- Checks agent tools (FFmpeg, Tesseract, Protoc, Whisper)
- Validates Ollama models (llama3.2, llava)
- Provides installation links for missing tools
- Color-coded output (green=pass, red=fail, yellow=warning)

**Usage:**
```batch
.\scripts\verify-setup.bat
# Or from root: verify-setup.bat
```

**Run this before setup to ensure all prerequisites are installed!**

### `compile-proto.bat`
Compiles protocol buffer definitions
- Generates Python gRPC code
- Updates imports automatically
- Run after modifying .proto files

**Usage:**
```batch
.\scripts\compile-proto.bat
```

### `download-models.bat`
Downloads required Ollama AI models
- Pulls llama3.2 (main reasoning)
- Pulls llava (vision analysis)
- Checks Ollama installation

**Usage:**
```batch
.\scripts\download-models.bat
```

## 📝 Quick Reference

**First time setup:**
```batch
# 1. Verify prerequisites
.\scripts\verify-setup.bat

# 2. Run setup
.\scripts\setup-all.bat

# 3. Download AI models
.\scripts\download-models.bat
```

**Daily usage:**
```batch
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start app
.\scripts\start-all.bat
```

**After code changes:**
```batch
# Recompile proto files
.\scripts\compile-proto.bat

# Reinstall dependencies
.\scripts\setup-all.bat
```

## 🎯 Script Locations

All scripts are now in the `/scripts` directory for better organization. The root directory has convenience launchers:

- `setup.bat` → redirects to `scripts\setup-all.bat`
- `start.bat` → redirects to `scripts\start-all.bat`

## 💡 Tips

1. **Always run scripts from project root**
   ```batch
   .\scripts\setup-all.bat  ✅ Correct
   cd scripts && setup-all.bat  ❌ Wrong
   ```

2. **Scripts use relative paths**
   - All paths are relative to project root
   - Works correctly when called from root directory

3. **Check for errors**
   - Scripts will pause on error
   - Read error messages carefully
   - Common fixes: reinstall dependencies, check prerequisites

4. **Need help?**
   - See `QUICKSTART.md` for detailed guide
   - See `README.md` for full documentation
   - Check troubleshooting sections

---

**Last Updated:** October 19, 2025
