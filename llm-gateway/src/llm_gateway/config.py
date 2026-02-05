"""Configuration for the LLM Gateway module.

Extends the shared configuration with gateway-specific settings.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoutingStrategy(str, Enum):
    """Available routing strategies."""

    FALLBACK = "fallback"
    ROUND_ROBIN = "round_robin"
    COST = "cost"
    LATENCY = "latency"


class GatewaySettings(BaseSettings):
    """LLM Gateway configuration settings.

    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="llm-gateway", description="Application name")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format: json or text")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Provider API Keys
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key",
    )

    # Provider URLs (optional overrides)
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL",
    )

    # Default models
    openai_default_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Default OpenAI model",
    )
    anthropic_default_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Default Anthropic model",
    )

    # Redis
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for distributed features",
    )

    # Routing
    default_routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.FALLBACK,
        description="Default routing strategy",
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Requests per minute per key",
    )
    rate_limit_tokens_per_minute: int = Field(
        default=100000,
        description="Tokens per minute per key",
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable semantic caching",
    )
    cache_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )
    cache_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for cache embeddings",
    )

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Failures before opening circuit",
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=30,
        description="Seconds before attempting recovery",
    )
    circuit_breaker_half_open_requests: int = Field(
        default=3,
        description="Requests to allow in half-open state",
    )

    # Retry
    retry_max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts",
    )
    retry_base_delay: float = Field(
        default=1.0,
        description="Base delay for exponential backoff (seconds)",
    )
    retry_max_delay: float = Field(
        default=10.0,
        description="Maximum delay between retries (seconds)",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        description="Exponential backoff base",
    )

    # Timeouts
    request_timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
    )
    connect_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds",
    )

    @property
    def providers_configured(self) -> list[str]:
        """Get list of configured providers."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        return providers


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    """Get cached gateway settings instance.

    Returns:
        GatewaySettings instance with loaded configuration.
    """
    return GatewaySettings()
