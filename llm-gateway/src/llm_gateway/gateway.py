"""Main gateway orchestrator.

Coordinates all components to handle LLM requests with
routing, caching, and resilience.
"""

import logging
import time
from typing import Any, AsyncIterator

from llm_gateway.config import GatewaySettings, get_gateway_settings
from llm_gateway.providers.base import LLMProvider, ProviderError
from llm_gateway.providers.factory import ProviderFactory, get_provider_factory
from llm_gateway.providers.models import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ProviderInfo,
    ProviderHealth,
)
from llm_gateway.routing.router import Router
from llm_gateway.routing.health import HealthTracker
from llm_gateway.resilience.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.resilience.retry import RetryPolicy, retry_with_policy
from llm_gateway.resilience.rate_limiter import RateLimiter
from llm_gateway.cache.semantic_cache import SemanticCache
from llm_gateway.cache.embeddings import EmbeddingService
from llm_gateway.cache.storage import InMemoryCacheStorage


logger = logging.getLogger(__name__)


class LLMGateway:
    """Main gateway orchestrating LLM requests.

    Handles:
    - Provider routing and selection
    - Semantic caching
    - Circuit breaker and retry logic
    - Rate limiting
    - Health tracking
    """

    def __init__(
        self,
        settings: GatewaySettings | None = None,
        provider_factory: ProviderFactory | None = None,
        cache: SemanticCache | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Initialize the gateway.

        Args:
            settings: Gateway settings.
            provider_factory: Provider factory.
            cache: Semantic cache.
            rate_limiter: Rate limiter.
        """
        self._settings = settings or get_gateway_settings()
        self._provider_factory = provider_factory or get_provider_factory(self._settings)

        # Initialize health tracker
        self._health_tracker = HealthTracker(
            health_check_interval=30.0,
            unhealthy_threshold=0.5,
        )

        # Initialize router
        self._router = Router(
            provider_factory=self._provider_factory,
            health_tracker=self._health_tracker,
            settings=self._settings,
        )

        # Initialize circuit breakers
        self._circuit_breakers = CircuitBreakerRegistry(
            default_failure_threshold=self._settings.circuit_breaker_failure_threshold,
            default_recovery_timeout=self._settings.circuit_breaker_recovery_timeout,
            default_half_open_requests=self._settings.circuit_breaker_half_open_requests,
        )

        # Initialize retry policy
        self._retry_policy = RetryPolicy(
            max_attempts=self._settings.retry_max_attempts,
            base_delay=self._settings.retry_base_delay,
            max_delay=self._settings.retry_max_delay,
            exponential_base=self._settings.retry_exponential_base,
        )

        # Initialize cache
        if cache:
            self._cache = cache
        elif self._settings.cache_enabled and self._settings.openai_api_key:
            embedding_service = EmbeddingService(
                settings=self._settings,
                model=self._settings.cache_embedding_model,
            )
            self._cache = SemanticCache(
                storage=InMemoryCacheStorage(),
                embedding_service=embedding_service,
                similarity_threshold=self._settings.cache_similarity_threshold,
                ttl=self._settings.cache_ttl_seconds,
                enabled=True,
            )
        else:
            self._cache = SemanticCache(enabled=False)

        # Store rate limiter
        self._rate_limiter = rate_limiter

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._cache.enabled

    @property
    def rate_limiting_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._rate_limiter is not None

    async def complete(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion.

        Handles routing, caching, circuit breaker, and retry logic.

        Args:
            request: The completion request.

        Returns:
            CompletionResponse from the selected provider.

        Raises:
            ProviderError: If all providers fail.
        """
        start_time = time.perf_counter()

        # Check cache first
        cache_result = await self._cache.get(request)
        if cache_result.hit and cache_result.response:
            logger.info(
                f"Cache hit with similarity {cache_result.similarity:.3f}",
                extra={"request_id": request.request_id},
            )
            return cache_result.response

        # Route to provider
        provider = await self._router.route(request)
        if provider is None:
            raise ProviderError(
                message="No available provider for request",
                provider="gateway",
                status_code=503,
                retryable=True,
            )

        # Track failed providers for fallback
        failed_providers: list[str] = []
        last_error: Exception | None = None

        # Try providers with circuit breaker and retry
        while provider is not None:
            circuit_breaker = self._circuit_breakers.get(provider.name)

            try:
                # Execute with circuit breaker
                response = await circuit_breaker.call(
                    self._execute_with_retry,
                    provider,
                    request,
                )

                # Record success
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._health_tracker.record_success(provider.name, latency_ms)

                # Update response latency
                response.latency_ms = latency_ms

                # Cache the response
                cache_key = await self._cache.set(request, response)
                if cache_key:
                    response.cache_key = cache_key

                return response

            except Exception as e:
                last_error = e
                failed_providers.append(provider.name)
                self._health_tracker.record_failure(provider.name, str(e))

                logger.warning(
                    f"Provider {provider.name} failed: {e}",
                    extra={"request_id": request.request_id},
                )

                # Try next provider
                provider = await self._router.route_with_fallback(
                    request,
                    failed_providers=failed_providers,
                )

        # All providers failed
        if last_error:
            raise last_error

        raise ProviderError(
            message="All providers failed",
            provider="gateway",
            status_code=503,
            retryable=True,
        )

    async def _execute_with_retry(
        self,
        provider: LLMProvider,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Execute request with retry logic.

        Args:
            provider: The provider to use.
            request: The completion request.

        Returns:
            CompletionResponse from the provider.
        """
        result = await retry_with_policy(
            provider.complete,
            self._retry_policy,
            request,
        )

        if result.success:
            return result.result
        else:
            raise result.exception  # type: ignore

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            request: The completion request.

        Yields:
            StreamChunk objects as they become available.
        """
        # Route to provider
        provider = await self._router.route(request)
        if provider is None:
            raise ProviderError(
                message="No available provider for request",
                provider="gateway",
                status_code=503,
                retryable=True,
            )

        circuit_breaker = self._circuit_breakers.get(provider.name)

        try:
            # For streaming, we use the circuit breaker but not retry
            # (can't retry a partial stream)
            async def stream_wrapper() -> AsyncIterator[StreamChunk]:
                async for chunk in provider.stream(request):
                    yield chunk

            # Check circuit state before streaming
            if circuit_breaker.is_open:
                from llm_gateway.resilience.circuit_breaker import CircuitOpenError

                raise CircuitOpenError(
                    name=circuit_breaker.name,
                    time_until_retry=0,
                    failure_count=circuit_breaker.failure_count,
                )

            async for chunk in provider.stream(request):
                yield chunk

            # Record success after stream completes
            self._health_tracker.record_success(provider.name, 0)

        except Exception as e:
            self._health_tracker.record_failure(provider.name, str(e))
            raise

    async def get_providers(self) -> list[ProviderInfo]:
        """Get information about all available providers.

        Returns:
            List of ProviderInfo instances.
        """
        providers = self._provider_factory.get_all()
        result = []

        for provider in providers.values():
            health = self._health_tracker.get_health(provider.name)
            info = provider.get_info(health)
            result.append(info)

        return result

    async def get_provider_health(self, provider_id: str) -> ProviderHealth | None:
        """Get health status for a specific provider.

        Args:
            provider_id: Provider identifier.

        Returns:
            ProviderHealth or None if provider not found.
        """
        provider = self._provider_factory.get(provider_id)
        if provider is None:
            return None

        return self._health_tracker.get_health(provider_id)

    async def check_health(self) -> dict[str, ProviderHealth]:
        """Check health of all providers.

        Returns:
            Dictionary mapping provider names to health status.
        """
        return await self._health_tracker.check_all_health()

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        if not self._cache.enabled:
            return {"enabled": False}

        stats = await self._cache.get_stats()
        return {
            "enabled": True,
            "size": stats.size,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "evictions": stats.evictions,
        }

    async def clear_cache(self) -> None:
        """Clear the semantic cache."""
        await self._cache.clear()

    async def startup(self) -> None:
        """Initialize gateway on startup."""
        # Start health check background task
        await self._health_tracker.start_background_checks()

        # Initial health check
        await self.check_health()

        logger.info(
            "LLM Gateway started",
            extra={
                "providers": self._provider_factory.get_available(),
                "cache_enabled": self.cache_enabled,
                "rate_limiting_enabled": self.rate_limiting_enabled,
            },
        )

    async def shutdown(self) -> None:
        """Cleanup gateway on shutdown."""
        await self._health_tracker.stop_background_checks()
        logger.info("LLM Gateway shutdown complete")
