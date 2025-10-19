"""
Transcription Agent MCP Server
Handles audio extraction and speech-to-text transcription
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import subprocess
import os


class TranscriptionAgent:
    """Agent for audio transcription tasks"""
    
    def __init__(self):
        self.temp_dir = Path("./temp")
    
    async def extract_audio(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to video file
            output_path: Path for output audio file
            
        Returns:
            Result with audio file path
        """
        try:
            video_file = Path(video_path)
            if not video_file.exists():
                return {"error": f"Video file not found: {video_path}"}
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_file),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                str(output_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and output_file.exists():
                return {
                    "success": True,
                    "audio_path": str(output_file),
                    "size_bytes": output_file.stat().st_size
                }
            else:
                return {
                    "error": f"ffmpeg failed: {result.stderr}"
                }
                
        except Exception as e:
            return {"error": f"Audio extraction failed: {str(e)}"}
    
    async def transcribe_audio(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio using whisper.cpp or similar local model
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: en)
            
        Returns:
            Transcription result with text and timestamps
        """
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return {"error": f"Audio file not found: {audio_path}"}
            
            # Path to whisper.cpp executable
            script_dir = Path(__file__).parent
            whisper_exe = script_dir / "whisper.cpp" / "whisper-cli.exe"
            model_path = script_dir / "models" / "ggml-base.bin"
            
            # Check if whisper.cpp is installed
            if not whisper_exe.exists():
                return {
                    "error": "whisper.cpp not found",
                    "setup_instructions": [
                        "1. Download whisper.cpp from https://github.com/ggerganov/whisper.cpp/releases",
                        "2. Extract to mcp-servers/transcription-agent/whisper.cpp/",
                        "3. Ensure whisper-cli.exe is at mcp-servers/transcription-agent/whisper.cpp/whisper-cli.exe"
                    ]
                }
            
            if not model_path.exists():
                return {"error": f"Model not found at {model_path}"}
            
            # Run whisper.cpp with JSON output for structured results
            cmd = [
                str(whisper_exe),
                "-m", str(model_path),
                "-f", str(audio_file),
                "-l", language,
                "-oj",  # Output JSON
                "-t", "4",  # Use 4 threads
                "-np"  # No print progress
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Parse JSON output from whisper.cpp
                output_json = audio_file.parent / f"{audio_file.stem}.json"
                
                if output_json.exists():
                    with open(output_json, 'r', encoding='utf-8') as f:
                        whisper_result = json.load(f)
                    
                    # Extract full text and segments
                    full_text = whisper_result.get("transcription", "")
                    segments = []
                    
                    for segment in whisper_result.get("transcription", []):
                        if isinstance(segment, dict):
                            segments.append({
                                "start": segment.get("offsets", {}).get("from", 0) / 1000.0,
                                "end": segment.get("offsets", {}).get("to", 0) / 1000.0,
                                "text": segment.get("text", "").strip()
                            })
                    
                    # Clean up JSON file
                    output_json.unlink(missing_ok=True)
                    
                    return {
                        "success": True,
                        "text": full_text,
                        "language": language,
                        "segments": segments
                    }
                else:
                    # Fallback: parse text output
                    text_output = result.stdout.strip()
                    return {
                        "success": True,
                        "text": text_output,
                        "language": language,
                        "segments": [],
                        "note": "JSON output not available, returning text only"
                    }
            else:
                return {
                    "error": f"whisper.cpp failed: {result.stderr}",
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {"error": "Transcription timeout (>5 minutes)"}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def get_timestamps(self, audio_path: str) -> Dict[str, Any]:
        """Get timestamped segments from transcription"""
        # This would integrate with the transcription tool
        result = await self.transcribe_audio(audio_path)
        if "segments" in result:
            return {
                "success": True,
                "timestamps": result["segments"]
            }
        return result


async def main():
    """Main entry point for the MCP server"""
    agent = TranscriptionAgent()
    server = Server("transcription-agent")
    
    # Register tools
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="extract_audio",
                description="Extract audio track from video file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to the video file"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path for the output audio file"
                        }
                    },
                    "required": ["video_path", "output_path"]
                }
            ),
            Tool(
                name="transcribe_audio",
                description="Transcribe audio to text with timestamps",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code (e.g., 'en', 'es', 'fr')",
                            "default": "en"
                        }
                    },
                    "required": ["audio_path"]
                }
            ),
            Tool(
                name="get_timestamps",
                description="Get timestamped segments from audio transcription",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file"
                        }
                    },
                    "required": ["audio_path"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        if name == "extract_audio":
            result = await agent.extract_audio(
                arguments["video_path"],
                arguments["output_path"]
            )
        elif name == "transcribe_audio":
            result = await agent.transcribe_audio(
                arguments["audio_path"],
                arguments.get("language", "en")
            )
        elif name == "get_timestamps":
            result = await agent.get_timestamps(arguments["audio_path"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
