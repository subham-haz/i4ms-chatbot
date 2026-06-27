"""Centralized configuration loaded from environment variables."""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # --- Vector store (pgvector) ---
    pg_host: str = Field(default="localhost", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")
    pg_user: str = Field(default="postgres", alias="PG_USER")
    pg_password: str = Field(default="postgres", alias="PG_PASSWORD")
    pg_database: str = Field(default="ragdb", alias="PG_DATABASE")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    hybrid_alpha: float = Field(default=0.5, alias="HYBRID_ALPHA")  # vector vs bm25
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    # --- Langfuse observability ---
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )
    langfuse_enabled: bool = Field(default=True, alias="LANGFUSE_ENABLED")

    # --- App ---
    environment: Literal["dev", "staging", "prod"] = Field(
        default="dev", alias="ENVIRONMENT"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def pg_connection_string(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
