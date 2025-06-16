from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Keys - will be added later
    # OPENAI_API_KEY: str = "your_openai_key"
    FIREBASE_CREDENTIALS_PATH: str = "" # Default to empty string

    # Database - will be added later
    # QDRANT_URL: str = "localhost:6333"
    # QDRANT_API_KEY: str | None = None

    # App Settings - will be added later
    # MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    # CHUNK_SIZE: int = 1000
    # CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
