"""
Configuration management using Pydantic Settings
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Server Configuration
    GRPC_HOST: str = "localhost"
    GRPC_PORT: int = 50051
    UPLOAD_DIR: Path = Path("./uploads")
    TEMP_DIR: Path = Path("./temp")
    REPORTS_DIR: Path = Path("./reports")
    
    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_VISION_MODEL: str = "llava"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "app.log"
    
    # Processing Limits
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_VIDEO_DURATION_SECONDS: int = 600
    MAX_CONCURRENT_ANALYSES: int = 3
    
    @property
    def max_video_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024


# Global settings instance
settings = Settings()
