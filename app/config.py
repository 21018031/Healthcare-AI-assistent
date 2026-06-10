from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str | None = None
    huggingfacehub_api_token: str | None = None
    
    # Model Configuration
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Paths
    vector_db_dir: Path = Path("./vector_store")
    knowledge_base_dir: Path = Path("./data")
    
    # Collection & Logging
    collection_name: str = "healthcare_knowledge_base"
    log_level: str = "INFO"
    
    # RAG Parameters
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k: int = 5
    
    # Mode
    use_mock: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()