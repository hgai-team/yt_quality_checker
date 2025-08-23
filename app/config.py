from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # API Settings
    app_name: str = "YouTube Quality Checker"
    api_version: str = "v1"
    debug: bool = False

    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int
    postgres_db: str

    # Qdrant Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # External Services
    embedding_service_url: str

    # Google Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # YouTube
    youtube_api_key: Optional[str] = None

    # Analysis Thresholds
    duplicate_threshold: float = 0.30
    image_similarity_threshold: float = 0.92
    text_similarity_threshold: float = 0.85
    static_video_confidence_threshold: float = 0.8
    inactive_days: int = 10

    # Performance
    batch_size: int = 32
    max_concurrent_downloads: int = 5
    text_embedding_dimension: int = 1536
    image_embedding_dimension: int = 1152
    
    project_keywords: dict = {
        "gaming": ["minecraft", "valorant", "gameplay", "gaming", "game", "playthrough"],
        "cooking": ["recipe", "cooking", "food", "cuisine", "chef", "baking"],
        "tech": ["technology", "gadget", "review", "unboxing", "tech", "software"],
        "education": ["tutorial", "learn", "education", "course", "lesson", "teaching"],
        "entertainment": ["comedy", "funny", "entertainment", "vlog", "reaction", "challenge"]
    }

@lru_cache
def get_settings():
    return Settings()

@lru_cache
def get_sql_db_path():
    settings = get_settings()

    postgres_user = settings.postgres_user
    postgres_password = settings.postgres_password
    postgres_host = settings.postgres_host
    postgres_port = settings.postgres_port
    postgres_db = settings.postgres_db

    return f"postgresql+asyncpg://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
