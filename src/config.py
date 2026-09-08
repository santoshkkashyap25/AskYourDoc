"""Application configuration module using Pydantic Settings."""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment or .env file."""
    
    # Application Settings
    APP_NAME: str = "AskYourDoc"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    CACHE_DB_PATH: Path = BASE_DIR / "data" / "cache.db"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_LEVEL: str = "INFO"

    # Storage & Document Limits
    MAX_DOCUMENTS: int = 10
    MAX_TOTAL_STORAGE_MB: int = 25

    # Document Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Embeddings
    EMBEDDING_PROVIDER: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RETRIEVER_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.25

    # LLM Settings
    LLM_PROVIDER: Literal["mock", "local", "openai", "gemini", "ollama", "groq", "grok"] = "mock"
    LOCAL_HF_MODEL_NAME: str = "google/flan-t5-base"

    # API Keys & Endpoints
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"

    GROK_API_KEY: Optional[str] = None
    GROK_MODEL_NAME: str = "grok-beta"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3"

    # Cache Settings
    CACHE_TTL_SECONDS: Optional[int] = 86400  # 24 hours

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure all required local directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
