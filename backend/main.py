"""
Main entry point for the Video Analysis Backend Service
Runs gRPC server only (Tauri connects via gRPC)
"""
import asyncio
import sys
from pathlib import Path

from loguru import logger

from src.grpc_server.server import serve
from src.utils.config import settings
from src.utils.setup import check_dependencies


def setup_logging():
    """Configure logging with loguru"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    if settings.LOG_FILE:
        logger.add(
            settings.LOG_FILE,
            rotation="10 MB",
            retention="7 days",
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )


async def main():
    """Main entry point"""
    logger.info("Starting Video Analysis Backend Service (gRPC only)")
    logger.info("Tauri frontend connects via gRPC on port 50051")
    
    # Setup
    setup_logging()
    
    # Check dependencies (non-blocking)
    dependencies_ok = await check_dependencies()
    if not dependencies_ok:
        logger.warning("Some dependency checks failed. Backend will start anyway.")
        logger.warning("Please ensure Ollama is running for full functionality.")
    else:
        logger.info("All dependency checks passed")
    
    # Start gRPC server (runs in main thread)
    logger.info("Starting gRPC server on localhost:50051")
    await serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
