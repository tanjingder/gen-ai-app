"""
MCP Client for communicating with MCP servers via stdio
Uses the official MCP SDK to spawn and communicate with stdio servers
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from loguru import logger

from ..utils.config import settings


class MCPClient:
    """Client for interacting with stdio MCP servers"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.venv_python = self.project_root / "backend" / "venv" / "Scripts" / "python.exe"
        
        # Server configurations
        self.servers = {
            "transcription": {
                "script": self.project_root / "mcp-servers" / "transcription-agent" / "server.py",
                "session": None,
                "read": None,
                "write": None,
                "exit_stack": None
            },
            "vision": {
                "script": self.project_root / "mcp-servers" / "vision-agent" / "server.py",
                "session": None,
                "read": None,
                "write": None,
                "exit_stack": None
            },
            "report": {
                "script": self.project_root / "mcp-servers" / "report-agent" / "server.py",
                "session": None,
                "read": None,
                "write": None,
                "exit_stack": None
            }
        }
        
        logger.info("MCP Client initialized for stdio servers")
    
    async def connect_server(self, server_name: str) -> bool:
        """
        Connect to an MCP stdio server
        
        Args:
            server_name: Name of the server (transcription, vision, or report)
            
        Returns:
            True if connection successful
        """
        try:
            server_config = self.servers.get(server_name)
            if not server_config:
                logger.error(f"Unknown server: {server_name}")
                return False
            
            # Check if already connected
            if server_config["session"] is not None:
                return True
            
            script_path = server_config["script"]
            if not script_path.exists():
                logger.error(f"Server script not found: {script_path}")
                return False
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=str(self.venv_python),
                args=[str(script_path)],
                env=None
            )
            
            # Connect to the server
            from contextlib import AsyncExitStack
            exit_stack = AsyncExitStack()
            
            read, write = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            session = await exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize the session
            await session.initialize()
            
            # Store connection
            server_config["session"] = session
            server_config["read"] = read
            server_config["write"] = write
            server_config["exit_stack"] = exit_stack
            
            logger.info(f"Connected to {server_name} MCP server")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to connect to {server_name} server: {e}")
            return False
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name not in self.servers:
            return
        
        server_config = self.servers[server_name]
        
        try:
            logger.info(f"Disconnecting from {server_name} MCP server...")
            
            # Close the exit stack (this closes stdin/stdout and terminates process)
            if server_config.get("exit_stack"):
                try:
                    # Use asyncio.wait_for with timeout to prevent hanging
                    await asyncio.wait_for(
                        server_config["exit_stack"].aclose(),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for {server_name} cleanup, forcing termination")
                    # Force terminate the process if it exists
                    if "session" in server_config:
                        # Process will be forcefully terminated by Python's cleanup
                        pass
                except RuntimeError as e:
                    if "cancel scope" in str(e).lower():
                        logger.debug(f"Ignoring harmless cancel scope error during {server_name} cleanup")
                    else:
                        raise
            
            logger.info(f"Disconnected from {server_name} MCP server")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {server_name}: {e}")
            # Don't raise - allow other cleanups to proceed
        finally:
            # Always clear the server config
            self.servers[server_name] = {}
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server
        
        Args:
            server_name: Name of the server (transcription, vision, or report)
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Ensure connection
            if not await self.connect_server(server_name):
                return {"error": f"Failed to connect to {server_name} server"}
            
            server_config = self.servers[server_name]
            session = server_config["session"]
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            
            # Parse result
            if result.content:
                # MCP returns list of content items
                content_text = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_text += content_item.text
                
                # Try to parse as JSON
                import json
                try:
                    parsed_result = json.loads(content_text)
                    logger.info(f"Tool {tool_name} executed successfully on {server_name}")
                    return parsed_result
                except json.JSONDecodeError:
                    # Return as text if not JSON
                    return {"result": content_text}
            else:
                return {"result": "Tool executed successfully (no content returned)"}
                
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name} on {server_name}: {e}")
            return {"error": str(e)}
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        List available tools from an MCP server
        
        Args:
            server_name: Name of the server (transcription, vision, or report)
            
        Returns:
            List of available tools
        """
        try:
            # Ensure connection
            if not await self.connect_server(server_name):
                return []
            
            server_config = self.servers[server_name]
            session = server_config["session"]
            
            # List tools
            tools_list = await session.list_tools()
            
            # Convert to dict format
            tools = []
            if tools_list and hasattr(tools_list, 'tools'):
                for tool in tools_list.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    })
            
            logger.info(f"Listed {len(tools)} tools from {server_name}")
            return tools
                
        except Exception as e:
            logger.exception(f"Error listing tools from {server_name}: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup all MCP server connections"""
        logger.info("Cleaning up all MCP connections...")
        
        # Disconnect from all servers concurrently with timeout
        disconnect_tasks = [
            self.disconnect_server(server_name) 
            for server_name in list(self.servers.keys())
        ]
        
        if disconnect_tasks:
            try:
                # Give total of 15 seconds for all cleanups
                await asyncio.wait_for(
                    asyncio.gather(*disconnect_tasks, return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("Cleanup timeout reached, some agents may not have closed cleanly")
        
        self.servers.clear()
        logger.info("All MCP connections cleaned up")
    
    # Transcription Agent Methods
    async def extract_audio(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """Extract audio from video file"""
        return await self.call_tool(
            "transcription",
            "extract_audio",
            {"video_path": video_path, "output_path": output_path}
        )
    
    async def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio to text"""
        args = {"audio_path": audio_path}
        if language:
            args["language"] = language
        return await self.call_tool(
            "transcription",
            "transcribe_audio",
            args
        )
    
    # Vision Agent Methods
    async def extract_frames(self, video_path: str, interval_seconds: float = 1.0, output_dir: str = None) -> Dict[str, Any]:
        """Extract frames from video"""
        args = {"video_path": video_path, "interval_seconds": interval_seconds}
        if output_dir:
            args["output_dir"] = output_dir
        return await self.call_tool(
            "vision",
            "extract_frames",
            args
        )
    
    async def analyze_frame(self, frame_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single frame"""
        args = {"frame_path": frame_path}
        if prompt:
            args["prompt"] = prompt
        return await self.call_tool(
            "vision",
            "analyze_frame",
            args
        )
    
    async def detect_objects(self, frame_path: str) -> Dict[str, Any]:
        """Detect objects in a frame"""
        return await self.call_tool(
            "vision",
            "detect_objects",
            {"frame_path": frame_path}
        )
    
    async def extract_text(self, frame_path: str) -> Dict[str, Any]:
        """Extract text from a frame using OCR"""
        return await self.call_tool(
            "vision",
            "extract_text",
            {"frame_path": frame_path}
        )
    
    async def analyze_chart(self, frame_path: str) -> Dict[str, Any]:
        """Analyze charts and graphs in a frame"""
        return await self.call_tool(
            "vision",
            "analyze_chart",
            {"frame_path": frame_path}
        )
    
    # Report Agent Methods
    async def create_pdf_report(self, analysis_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Create PDF report from analysis data"""
        return await self.call_tool(
            "report",
            "create_pdf_report",
            {"content": analysis_data, "output_path": output_path}  # Changed from analysis_data to content
        )
    
    async def create_ppt_report(self, analysis_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Create PowerPoint report from analysis data"""
        return await self.call_tool(
            "report",
            "create_ppt_report",
            {"content": analysis_data, "output_path": output_path}  # Changed from analysis_data to content
        )
    
    async def create_text_report(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create text summary from analysis data"""
        return await self.call_tool(
            "report",
            "create_text_report",
            {"analysis_data": analysis_data}
        )


# Global MCP client instance
mcp_client = MCPClient()
