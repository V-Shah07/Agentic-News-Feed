"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    database_url: str = "postgresql+psycopg2://newsuser:newspass@localhost:5432/newsfeed"
    redis_url: str = "redis://localhost:6379/0"

    # Vector store (Phase 2+)
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: str = "./chroma_data"

    # Embeddings (Phase 2+)
    embedding_model: str = "all-MiniLM-L6-v2"
    # Path to a promoted fine-tuned model (Phase 6). Falls back to base when unset.
    embedding_model_path: str = ""

    # Novelty filter (Phase 2)
    # Calibrated from the empirical similarity distribution of a real ingested
    # batch (see logs/phase2_threshold_calibration.log): every cross-source pair
    # at cosine >= 0.70 was a verified same-story duplicate, while distinct
    # stories sat below it. Tunable to trade recall vs dedup aggressiveness.
    novelty_threshold: float = 0.70
    novelty_window_days: int = 30

    # LLM (Phase 4/5)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ingestion
    ingest_interval_seconds: int = 3600
    ingest_per_source_limit: int = 20

    # MLflow (Phase 6) — SQLite backend so the model registry is available.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
