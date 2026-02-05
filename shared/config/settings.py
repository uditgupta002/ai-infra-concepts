"""Centralized configuration management using Pydantic Settings.

Provides typed, validated configuration with support for:
- Environment variable loading
- .env file support
- Hierarchical configuration (base -> module-specific)
- Runtime validation
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log level configuration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseSettings(PydanticBaseSettings):
    """Base settings class for all modules.

    Provides common configuration options and environment variable loading.
    Subclass this for module-specific settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application metadata
    app_name: str = Field(default="ai-infra", description="Application name")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Logging configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'text'",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Redis configuration (optional, for caching/rate limiting)
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "BaseSettings":
        """Validate settings after loading."""
        # Enable debug logging in development if debug is True
        if self.debug and self.log_level != LogLevel.DEBUG:
            object.__setattr__(self, "log_level", LogLevel.DEBUG)
        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING


@lru_cache
def get_settings() -> BaseSettings:
    """Get cached settings instance.

    Uses LRU cache to ensure singleton behavior.
    Call get_settings.cache_clear() to reload settings.

    Returns:
        BaseSettings instance with loaded configuration.
    """
    return BaseSettings()


def load_settings_from_file(
    settings_class: type[BaseSettings],
    config_path: Path | str,
) -> BaseSettings:
    """Load settings from a specific configuration file.

    Args:
        settings_class: The settings class to instantiate.
        config_path: Path to the configuration file.

    Returns:
        Settings instance loaded from the file.
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Create settings with custom env file
    return settings_class(_env_file=config_path)
