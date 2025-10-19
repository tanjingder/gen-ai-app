"""
Test script for Transcription Agent
Run this to verify the agent works independently of the MCP server
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import the agent
sys.path.insert(0, str(Path(__file__).parent))

from server import TranscriptionAgent


async def test_agent():
    """Test the transcription agent functionality"""
    print("=" * 60)
    print("TRANSCRIPTION AGENT TEST")
    print("=" * 60)
    
    agent = TranscriptionAgent()
    
    # Test 1: Check whisper.cpp installation
    print("\n[Test 1] Checking whisper.cpp installation...")
    script_dir = Path(__file__).parent
    whisper_exe = script_dir / "whisper.cpp" / "main.exe"
    model_path = script_dir / "models" / "ggml-base.bin"
    
    print(f"  Whisper executable: {whisper_exe}")
    print(f"  Exists: {whisper_exe.exists()} {'✅' if whisper_exe.exists() else '❌'}")
    
    print(f"  Model file: {model_path}")
    print(f"  Exists: {model_path.exists()} {'✅' if model_path.exists() else '❌'}")
    
    if not whisper_exe.exists():
        print("\n⚠️  SETUP REQUIRED:")
        print("  1. Download whisper.cpp from: https://github.com/ggerganov/whisper.cpp/releases")
        print("  2. Extract to: mcp-servers/transcription-agent/whisper.cpp/")
        print("  3. Ensure main.exe is present")
        print("\n  See WHISPER_SETUP.md for detailed instructions")
        return
    
    if not model_path.exists():
        print("\n❌ Model file not found!")
        print("  Please place ggml-base.bin in the models/ directory")
        return
    
    print("\n✅ All files present!")
    
    # Test 2: Create a test audio file (optional)
    print("\n[Test 2] Testing audio extraction...")
    print("  Note: This requires a video file to test")
    print("  Skipping for now (requires actual video file)")
    
    # Test 3: Test transcription with a sample
    print("\n[Test 3] Testing transcription setup...")
    # We'll test the method exists and handles errors properly
    test_audio = Path("nonexistent.wav")
    result = await agent.transcribe_audio(str(test_audio))
    
    if "error" in result:
        if "not found" in result["error"].lower():
            print("  ✅ Error handling works (file not found detected)")
        else:
            print(f"  ⚠️  Unexpected error: {result['error']}")
    else:
        print("  ✅ Transcription method responds correctly")
    
    # Test 4: If you have a test audio file, uncomment this
    print("\n[Test 4] Real transcription test...")
    print("  To test with real audio:")
    print("  1. Place a .wav file in the temp/ directory")
    print("  2. Update the path below and uncomment the code")
    print("  ")
    print("  Example:")
    print('    test_audio = Path("temp/test.wav")')
    print('    if test_audio.exists():')
    print('        result = await agent.transcribe_audio(str(test_audio))')
    print('        print(result)')
    
    # Uncomment to test with real audio:
    real_audio = script_dir / "temp" / "test.wav"
    if real_audio.exists():
        print(f"\n  Found test audio: {real_audio}")
        print("  Running transcription...")
        result = await agent.transcribe_audio(str(real_audio))
        print("\n  Result:")
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"  No test audio found at: {real_audio}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    if whisper_exe.exists() and model_path.exists():
        print("\n✅ Transcription agent is ready to use!")
        print("\nNext steps:")
        print("  1. Start the MCP server: start-mcp-servers.bat")
        print("  2. The agent will be available on port 8001")
        print("  3. Backend orchestrator can call it via MCP protocol")
    else:
        print("\n⚠️  Setup incomplete - see messages above")


if __name__ == "__main__":
    print("\nRunning transcription agent test...")
    print("This tests the agent logic without the MCP server protocol\n")
    
    try:
        asyncio.run(test_agent())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
