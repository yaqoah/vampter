from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Configuration management system for the Vampter backend.
    Loads and validates configuration from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # FastAPI Server configuration
    env: str = Field("development", validation_alias="APP_ENV")
    debug: bool = Field(True, validation_alias="APP_DEBUG")
    host: str = Field("0.0.0.0", validation_alias="APP_HOST")
    port: int = Field(8000, validation_alias="APP_PORT")

    # Qdrant Vector DB configuration
    qdrant_host: str = Field("localhost", validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, validation_alias="QDRANT_PORT")
    qdrant_port_grpc: int = Field(6334, validation_alias="QDRANT_PORT_GRPC")
    qdrant_api_key: Optional[SecretStr] = Field(None, validation_alias="QDRANT_API_KEY")

    # Neo4j Graph DB configuration
    neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", validation_alias="NEO4J_USER")
    neo4j_password: SecretStr = Field("vampter_neo4j_password", validation_alias="NEO4J_PASSWORD")

    # Redis Cache configuration
    redis_host: str = Field("localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(6379, validation_alias="REDIS_PORT")
    redis_db: int = Field(0, validation_alias="REDIS_DB")
    redis_password: Optional[SecretStr] = Field(None, validation_alias="REDIS_PASSWORD")

    # Mistral AI LLM configuration
    mistral_api_key: Optional[SecretStr] = Field(None, validation_alias="MISTRAL_API_KEY")
    llm_model: str = Field("mistral-large-latest", validation_alias="LLM_MODEL")

    @property
    def redis_url(self) -> str:
        """
        Constructs the Redis connection URL.
        """
        if self.redis_password and self.redis_password.get_secret_value():
            return f"redis://:{self.redis_password.get_secret_value()}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

# Instantiated settings container
settings = AppSettings()
