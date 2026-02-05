"""API request and response schemas.

Defines Pydantic models for API validation and documentation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llm_gateway.providers.models import (
    Message,
    MessageRole,
    TokenUsage,
    CostEstimate,
    ProviderHealth,
    RoutingHints,
)


class CompletionRequestSchema(BaseModel):
    """API schema for completion requests."""

    messages: list[Message] = Field(
        min_length=1,
        description="Conversation messages",
        json_schema_extra={
            "examples": [
                [{"role": "user", "content": "Hello, how are you?"}]
            ]
        },
    )
    model: str | None = Field(
        default=None,
        description="Specific model to use",
    )
    provider: str | None = Field(
        default=None,
        description="Specific provider to use",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Stop sequences",
    )
    routing: RoutingHints = Field(
        default_factory=RoutingHints,
        description="Routing preferences",
    )
    skip_cache: bool = Field(
        default=False,
        description="Skip cache for this request",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for rate limiting",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
            ]
        }
    }


class CompletionResponseSchema(BaseModel):
    """API schema for completion responses."""

    id: str = Field(description="Response ID")
    content: str = Field(description="Generated content")
    model: str = Field(description="Model used")
    provider: str = Field(description="Provider used")
    usage: TokenUsage = Field(description="Token usage")
    cost: CostEstimate | None = Field(default=None, description="Estimated cost")
    latency_ms: float = Field(description="Request latency in milliseconds")
    cached: bool = Field(default=False, description="Whether response was cached")
    request_id: str = Field(description="Original request ID")
    timestamp: datetime = Field(description="Response timestamp")
    finish_reason: str | None = Field(default=None, description="Finish reason")


class StreamChunkSchema(BaseModel):
    """API schema for streaming chunks."""

    id: str = Field(description="Chunk ID")
    content: str = Field(description="Content delta")
    model: str = Field(description="Model name")
    provider: str = Field(description="Provider name")
    finish_reason: str | None = Field(default=None, description="Finish reason")
    usage: TokenUsage | None = Field(default=None, description="Usage (final chunk only)")


class ProviderInfoSchema(BaseModel):
    """API schema for provider information."""

    name: str = Field(description="Provider identifier")
    display_name: str = Field(description="Display name")
    is_available: bool = Field(description="Whether provider is available")
    supported_models: list[str] = Field(description="Supported models")
    default_model: str = Field(description="Default model")
    capabilities: list[str] = Field(description="Provider capabilities")
    health: ProviderHealth = Field(description="Health status")


class ProviderListResponse(BaseModel):
    """API response for listing providers."""

    providers: list[ProviderInfoSchema] = Field(description="Available providers")
    total: int = Field(description="Total number of providers")


class HealthResponse(BaseModel):
    """API schema for health check response."""

    status: str = Field(description="Overall health status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="Check timestamp")
    providers: dict[str, ProviderHealth] = Field(description="Provider health")
    cache_enabled: bool = Field(description="Whether caching is enabled")
    rate_limiting_enabled: bool = Field(description="Whether rate limiting is enabled")


class CacheStatsResponse(BaseModel):
    """API schema for cache statistics."""

    enabled: bool = Field(description="Whether caching is enabled")
    size: int = Field(description="Number of cached entries")
    hits: int = Field(description="Cache hits")
    misses: int = Field(description="Cache misses")
    hit_rate: float = Field(description="Cache hit rate")
    evictions: int = Field(description="Cache evictions")


class ErrorResponse(BaseModel):
    """API schema for error responses."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    request_id: str | None = Field(default=None, description="Request ID")
    details: dict[str, Any] = Field(default_factory=dict, description="Error details")
