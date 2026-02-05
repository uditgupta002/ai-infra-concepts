"""Tests for the main gateway."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from llm_gateway.gateway import LLMGateway
from llm_gateway.providers.models import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    TokenUsage,
)
from llm_gateway.providers.factory import ProviderFactory
from llm_gateway.providers.base import ProviderError
from llm_gateway.cache.semantic_cache import SemanticCache
from tests.conftest import MockProvider


class TestLLMGateway:
    """Tests for LLMGateway."""

    @pytest.fixture
    def mock_factory(self):
        """Create a mock provider factory."""
        factory = MagicMock(spec=ProviderFactory)
        provider = MockProvider()
        factory.get.return_value = provider
        factory.get_all.return_value = {"mock": provider}
        factory.get_available.return_value = ["mock"]
        return factory

    @pytest.fixture
    def gateway(self, mock_factory):
        """Create a gateway with mock factory."""
        return LLMGateway(
            provider_factory=mock_factory,
            cache=SemanticCache(enabled=False),
        )

    @pytest.mark.asyncio
    async def test_complete_success(self, gateway, sample_request):
        """Should complete request successfully."""
        response = await gateway.complete(sample_request)

        assert response is not None
        assert response.content == "Mock response"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_complete_with_provider_failure_fallback(self):
        """Should fallback when provider fails."""
        # Create providers
        failing_provider = MockProvider(name="failing", should_fail=True)
        working_provider = MockProvider(name="working")

        factory = MagicMock(spec=ProviderFactory)
        factory.get_all.return_value = {
            "failing": failing_provider,
            "working": working_provider,
        }
        factory.get.side_effect = lambda name: {
            "failing": failing_provider,
            "working": working_provider,
        }.get(name)
        factory.get_available.return_value = ["failing", "working"]

        gateway = LLMGateway(
            provider_factory=factory,
            cache=SemanticCache(enabled=False),
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        # Should succeed with fallback to working provider
        response = await gateway.complete(request)
        assert response.provider == "working"

    @pytest.mark.asyncio
    async def test_complete_all_providers_fail(self):
        """Should raise error when all providers fail."""
        provider = MockProvider(name="failing", should_fail=True)

        factory = MagicMock(spec=ProviderFactory)
        factory.get_all.return_value = {"failing": provider}
        factory.get.return_value = provider
        factory.get_available.return_value = ["failing"]

        gateway = LLMGateway(
            provider_factory=factory,
            cache=SemanticCache(enabled=False),
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        with pytest.raises(ProviderError):
            await gateway.complete(request)

    @pytest.mark.asyncio
    async def test_stream_success(self, gateway):
        """Should stream response successfully."""
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        chunks = []
        async for chunk in gateway.stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("Mock" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_get_providers(self, gateway):
        """Should return provider information."""
        providers = await gateway.get_providers()

        assert len(providers) == 1
        assert providers[0].name == "mock"

    @pytest.mark.asyncio
    async def test_check_health(self, gateway):
        """Should check provider health."""
        health = await gateway.check_health()

        assert "mock" in health
        assert health["mock"].is_healthy

    @pytest.mark.asyncio
    async def test_cache_stats(self, gateway):
        """Should return cache statistics."""
        stats = await gateway.get_cache_stats()

        assert "enabled" in stats
        assert stats["enabled"] is False

    @pytest.mark.asyncio
    async def test_clear_cache(self, gateway):
        """Should clear cache without error."""
        await gateway.clear_cache()


class TestGatewayWithCache:
    """Tests for gateway with caching enabled."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Should return cached response on cache hit."""
        from llm_gateway.cache.embeddings import MockEmbeddingService
        from llm_gateway.cache.storage import InMemoryCacheStorage

        provider = MockProvider()
        factory = MagicMock(spec=ProviderFactory)
        factory.get_all.return_value = {"mock": provider}
        factory.get.return_value = provider
        factory.get_available.return_value = ["mock"]

        cache = SemanticCache(
            storage=InMemoryCacheStorage(),
            embedding_service=MockEmbeddingService(),
            similarity_threshold=0.95,
        )

        gateway = LLMGateway(
            provider_factory=factory,
            cache=cache,
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test1",
        )

        # First call - cache miss
        response1 = await gateway.complete(request)
        call_count_after_first = provider._call_count

        # Second call with same content - should be cache hit
        request2 = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test2",
        )
        response2 = await gateway.complete(request2)

        # Provider should not be called again
        assert provider._call_count == call_count_after_first
        assert response2.cached


class TestGatewayLifecycle:
    """Tests for gateway lifecycle methods."""

    @pytest.mark.asyncio
    async def test_startup_shutdown(self):
        """Should handle startup and shutdown gracefully."""
        provider = MockProvider()
        factory = MagicMock(spec=ProviderFactory)
        factory.get_all.return_value = {"mock": provider}
        factory.get_available.return_value = ["mock"]

        gateway = LLMGateway(
            provider_factory=factory,
            cache=SemanticCache(enabled=False),
        )

        await gateway.startup()
        assert gateway.cache_enabled is False

        await gateway.shutdown()
