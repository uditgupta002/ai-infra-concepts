"""Abstract base class for LLM providers.

Defines the contract that all LLM provider implementations must follow.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator

from llm_gateway.providers.models import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ProviderHealth,
    ProviderInfo,
    CostEstimate,
    TokenUsage,
)


class ProviderCapability(str, Enum):
    """Capabilities that a provider may support."""

    CHAT = "chat"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    JSON_MODE = "json_mode"
    EMBEDDINGS = "embeddings"


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize provider error.

        Args:
            message: Error message.
            provider: Provider name.
            status_code: HTTP status code if applicable.
            retryable: Whether the error is retryable.
            original_error: Original exception if wrapping.
        """
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.original_error = original_error

    def __str__(self) -> str:
        return f"[{self.provider}] {super().__str__()}"


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message=message,
            provider=provider,
            status_code=429,
            retryable=True,
        )
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Authentication failed error."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(
            message=message,
            provider=provider,
            status_code=401,
            retryable=False,
        )


class InvalidRequestError(ProviderError):
    """Invalid request error."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(
            message=message,
            provider=provider,
            status_code=400,
            retryable=False,
        )


class ServiceUnavailableError(ProviderError):
    """Service unavailable error."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(
            message=message,
            provider=provider,
            status_code=503,
            retryable=True,
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class and
    implement the required abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider.

        Returns:
            Provider name (e.g., 'openai', 'anthropic').
        """
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name.

        Returns:
            Display name (e.g., 'OpenAI', 'Anthropic').
        """
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of supported model identifiers.

        Returns:
            List of model names.
        """
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use if none specified.

        Returns:
            Default model name.
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> list[ProviderCapability]:
        """List of capabilities this provider supports.

        Returns:
            List of ProviderCapability values.
        """
        ...

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion for the given request.

        Args:
            request: The completion request.

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ProviderError: If the request fails.
        """
        ...

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            request: The completion request.

        Yields:
            StreamChunk objects as they become available.

        Raises:
            ProviderError: If the request fails.
        """
        ...

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check the health of this provider.

        Returns:
            ProviderHealth status.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in the given text.

        Args:
            text: Text to count tokens in.
            model: Optional model to use for counting (uses default if not set).

        Returns:
            Number of tokens.
        """
        ...

    @abstractmethod
    def estimate_cost(
        self,
        request: CompletionRequest,
        estimated_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Estimate the cost of a request.

        Args:
            request: The completion request.
            estimated_output_tokens: Optional estimated output tokens.

        Returns:
            CostEstimate for the request.
        """
        ...

    def get_model(self, request: CompletionRequest) -> str:
        """Get the model to use for a request.

        Args:
            request: The completion request.

        Returns:
            Model name to use.
        """
        if request.model and request.model in self.supported_models:
            return request.model
        return self.default_model

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a specific model.

        Args:
            model: Model name to check.

        Returns:
            True if the model is supported.
        """
        return model in self.supported_models

    def has_capability(self, capability: ProviderCapability) -> bool:
        """Check if this provider has a specific capability.

        Args:
            capability: Capability to check.

        Returns:
            True if the capability is supported.
        """
        return capability in self.capabilities

    def get_info(self, health: ProviderHealth | None = None) -> ProviderInfo:
        """Get information about this provider.

        Args:
            health: Optional pre-fetched health status.

        Returns:
            ProviderInfo instance.
        """
        return ProviderInfo(
            name=self.name,
            display_name=self.display_name,
            is_available=True,  # Will be updated based on health
            supported_models=self.supported_models,
            default_model=self.default_model,
            capabilities=[cap.value for cap in self.capabilities],
            health=health or ProviderHealth(is_healthy=True),
            pricing=self._get_pricing(),
        )

    def _get_pricing(self) -> dict[str, dict[str, float]]:
        """Get pricing information for models.

        Override in subclass to provide actual pricing.

        Returns:
            Dict mapping model names to pricing info.
        """
        return {}

    def _calculate_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """Calculate token usage.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            TokenUsage instance.
        """
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
