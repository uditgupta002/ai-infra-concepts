"""Provider-agnostic models for LLM interactions.

These models provide a unified interface for requests and responses
across all LLM providers.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class MessageRole(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Content of the message")
    name: str | None = Field(default=None, description="Optional name for the sender")
    tool_call_id: str | None = Field(
        default=None,
        description="Tool call ID for tool responses",
    )

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        result: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic API format."""
        return {
            "role": self.role.value if self.role != MessageRole.SYSTEM else "user",
            "content": self.content,
        }


class TokenUsage(BaseModel):
    """Token usage information for a completion."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")

    @classmethod
    def zero(cls) -> "TokenUsage":
        """Create zero usage instance."""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class CostEstimate(BaseModel):
    """Cost estimate for a completion request."""

    model_config = ConfigDict(extra="ignore")

    input_cost: float = Field(description="Cost for input tokens in USD")
    output_cost: float = Field(description="Cost for output tokens in USD")
    total_cost: float = Field(description="Total estimated cost in USD")
    currency: str = Field(default="USD", description="Currency code")

    @classmethod
    def from_usage(
        cls,
        usage: TokenUsage,
        input_price_per_1k: float,
        output_price_per_1k: float,
    ) -> "CostEstimate":
        """Calculate cost from token usage.

        Args:
            usage: Token usage information.
            input_price_per_1k: Price per 1000 input tokens.
            output_price_per_1k: Price per 1000 output tokens.

        Returns:
            CostEstimate instance.
        """
        input_cost = (usage.prompt_tokens / 1000) * input_price_per_1k
        output_cost = (usage.completion_tokens / 1000) * output_price_per_1k
        return cls(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )


class RoutingHints(BaseModel):
    """Hints for request routing."""

    model_config = ConfigDict(extra="ignore")

    strategy: str | None = Field(
        default=None,
        description="Routing strategy to use",
    )
    preferred_providers: list[str] = Field(
        default_factory=list,
        description="Preferred provider order",
    )
    exclude_providers: list[str] = Field(
        default_factory=list,
        description="Providers to exclude",
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Required provider capabilities",
    )


class CompletionRequest(BaseModel):
    """Request for a chat completion."""

    model_config = ConfigDict(extra="forbid")

    # Required
    messages: list[Message] = Field(
        min_length=1,
        description="Messages in the conversation",
    )

    # Model selection
    model: str | None = Field(
        default=None,
        description="Specific model to use (provider will use default if not set)",
    )
    provider: str | None = Field(
        default=None,
        description="Specific provider to use",
    )

    # Generation parameters
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
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty",
    )

    # Routing
    routing: RoutingHints = Field(
        default_factory=RoutingHints,
        description="Routing hints",
    )

    # Request metadata
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier",
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier for rate limiting",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )

    # Cache control
    skip_cache: bool = Field(
        default=False,
        description="Skip cache lookup for this request",
    )

    def get_cache_key_content(self) -> str:
        """Get content for cache key generation.

        Returns normalized message content for embedding-based caching.
        """
        parts = []
        for msg in self.messages:
            parts.append(f"{msg.role.value}: {msg.content}")

        # Include relevant generation parameters in cache key
        params = f"temp={self.temperature}"
        if self.max_tokens:
            params += f",max_tokens={self.max_tokens}"
        if self.model:
            params += f",model={self.model}"

        return "\n".join(parts) + f"\n[{params}]"


class CompletionResponse(BaseModel):
    """Response from a chat completion."""

    model_config = ConfigDict(extra="ignore")

    # Core response
    id: str = Field(description="Unique response identifier")
    content: str = Field(description="Generated content")

    # Provider information
    model: str = Field(description="Model that generated the response")
    provider: str = Field(description="Provider that handled the request")

    # Usage and cost
    usage: TokenUsage = Field(description="Token usage")
    cost: CostEstimate | None = Field(
        default=None,
        description="Estimated cost",
    )

    # Performance
    latency_ms: float = Field(description="Request latency in milliseconds")

    # Caching
    cached: bool = Field(default=False, description="Whether response was cached")
    cache_key: str | None = Field(
        default=None,
        description="Cache key if caching was attempted",
    )

    # Metadata
    request_id: str = Field(description="Original request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )

    # Finish reason
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion (stop, length, etc.)",
    )


class StreamChunk(BaseModel):
    """A chunk of a streaming response."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Chunk identifier")
    content: str = Field(description="Content delta")
    model: str = Field(description="Model name")
    provider: str = Field(description="Provider name")
    finish_reason: str | None = Field(
        default=None,
        description="Finish reason (only in final chunk)",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Usage (only in final chunk)",
    )

    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk."""
        return self.finish_reason is not None


class ProviderHealth(BaseModel):
    """Health status for a provider."""

    model_config = ConfigDict(extra="ignore")

    is_healthy: bool = Field(description="Whether provider is healthy")
    latency_ms: float | None = Field(
        default=None,
        description="Latest latency measurement",
    )
    latency_p50_ms: float | None = Field(
        default=None,
        description="50th percentile latency",
    )
    latency_p95_ms: float | None = Field(
        default=None,
        description="95th percentile latency",
    )
    latency_p99_ms: float | None = Field(
        default=None,
        description="99th percentile latency",
    )
    success_rate: float = Field(
        default=1.0,
        description="Success rate (0-1)",
    )
    error_count: int = Field(
        default=0,
        description="Recent error count",
    )
    last_error: str | None = Field(
        default=None,
        description="Last error message",
    )
    last_check: datetime | None = Field(
        default=None,
        description="Last health check time",
    )


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Provider name")
    display_name: str = Field(description="Display name")
    is_available: bool = Field(description="Whether provider is available")
    supported_models: list[str] = Field(description="Supported model names")
    default_model: str = Field(description="Default model")
    capabilities: list[str] = Field(description="Provider capabilities")
    health: ProviderHealth = Field(description="Health status")
    pricing: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Pricing per model (input/output per 1k tokens)",
    )
