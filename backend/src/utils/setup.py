"""
Setup utilities for initializing the application
"""
import httpx
from pathlib import Path
from loguru import logger

from .config import settings


async def check_ollama_connection() -> bool:
    """Check if Ollama is accessible"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Ollama is running. Available models: {[m['name'] for m in models]}")
                
                # Check if required models are available
                model_names = [m["name"] for m in models]
                required_models = [settings.OLLAMA_MODEL, settings.OLLAMA_VISION_MODEL]
                
                missing_models = [m for m in required_models if not any(m in name for name in model_names)]
                
                if missing_models:
                    logger.warning(f"Missing models: {missing_models}. Run: ollama pull {' '.join(missing_models)}")
                    logger.warning("Backend will start anyway, but video analysis features will not work until models are downloaded.")
                
                return True  # Return True even if models are missing - Ollama is running
            return False
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.warning("Backend will start anyway, but video analysis features will not work until Ollama is running.")
        return True  # Return True anyway to allow backend to start


async def check_mcp_server(name: str, url: str) -> bool:
    """Check if an MCP server is accessible"""
    try:
        async with httpx.AsyncClient() as client:
            # Try to connect to the MCP server health endpoint
            response = await client.get(f"{url}/health", timeout=3.0)
            if response.status_code == 200:
                logger.info(f"{name} MCP server is running at {url}")
                return True
            return False
    except Exception as e:
        logger.warning(f"{name} MCP server not accessible at {url}: {e}")
        return False


async def check_dependencies() -> bool:
    """Check all external dependencies"""
    logger.info("Checking dependencies...")
    
    # Check Ollama
    ollama_ok = await check_ollama_connection()
    if not ollama_ok:
        logger.error("Ollama is not running or required models are not available")
        return False
    
    # Note: MCP servers are now stdio-based and spawned on-demand by MCPClient
    # No need to check HTTP endpoints
    logger.info("MCP servers will be spawned on-demand via stdio")
    
    return True
