from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # OpenAI Settings
    OPENAI_API_KEY: str = "" # Will be loaded from .env
    FIREBASE_CREDENTIALS_PATH: str = "" # Default to empty string

    # Qdrant Settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "rag_documents"

    # Relational Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://user:password@host:port/dbname" # Placeholder

    # Text Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    # MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB # Still commented out as per original, can be activated later

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
