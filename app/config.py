"""Application Configuration and Environment Settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "DocuMind"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # LLM Settings
    LLM_PROVIDER: str = "mock"  # "gemini", "ollama", or "mock"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:latest"

    # Chunking & Retrieval
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Grounding guard: refuse to answer when the best cosine similarity is below
    # this floor. Required by the sample policy (Section 4.2) and honoured here.
    MIN_SIMILARITY_SCORE: float = 0.35

    # Uploads. A single allow-list shared by the API layer and the document loader
    # so the two contracts cannot drift apart.
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".csv"]

    # Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    STATIC_DIR: Path = BASE_DIR / "static"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vector_db"

    # Host & Port
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # CORS. Empty by default: the SPA is served from the same origin, so no
    # cross-origin access is required and no wildcard is granted implicitly.
    # Set e.g. CORS_ORIGINS=["http://localhost:5173"] to allow a separate frontend.
    CORS_ORIGINS: list[str] = []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
