"""Tests for routing strategies."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_gateway.routing.strategies import (
    RoutingStrategy,
    FallbackChainStrategy,
    RoundRobinStrategy,
    CostBasedStrategy,
    LatencyBasedStrategy,
    WeightedRandomStrategy,
    RoutingContext,
    ProviderMetrics,
    get_strategy,
)
from llm_gateway.providers.models import CompletionRequest, Message, MessageRole, CostEstimate


class TestFallbackChainStrategy:
    """Tests for FallbackChainStrategy."""

    @pytest.mark.asyncio
    async def test_select_first_available_provider(self, mock_provider):
        """Should select first available provider in priority order."""
        strategy = FallbackChainStrategy(priority_order=["mock", "other"])

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[mock_provider],
        )

        selected = await strategy.select_provider(context)
        assert selected is not None
        assert selected.name == "mock"

    @pytest.mark.asyncio
    async def test_skip_excluded_providers(self, mock_provider):
        """Should skip providers in exclusion list."""
        strategy = FallbackChainStrategy()

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[mock_provider],
            excluded_providers=["mock"],
        )

        selected = await strategy.select_provider(context)
        assert selected is None

    @pytest.mark.asyncio
    async def test_skip_unhealthy_providers(self, mock_provider):
        """Should skip unhealthy providers."""
        strategy = FallbackChainStrategy()

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[mock_provider],
            provider_metrics={"mock": ProviderMetrics(name="mock", is_healthy=False)},
        )

        selected = await strategy.select_provider(context)
        assert selected is None

    @pytest.mark.asyncio
    async def test_respect_preferences(self, mock_provider):
        """Should prioritize preferred providers."""
        strategy = FallbackChainStrategy()

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[mock_provider],
            preferred_providers=["mock"],
        )

        selected = await strategy.select_provider(context)
        assert selected is not None
        assert selected.name == "mock"


class TestRoundRobinStrategy:
    """Tests for RoundRobinStrategy."""

    @pytest.mark.asyncio
    async def test_rotates_through_providers(self):
        """Should rotate through providers."""
        from tests.conftest import MockProvider

        strategy = RoundRobinStrategy()
        providers = [
            MockProvider(name="provider1"),
            MockProvider(name="provider2"),
            MockProvider(name="provider3"),
        ]

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="test")],
            request_id="test",
        )

        selections = []
        for _ in range(6):
            context = RoutingContext(
                request=request,
                available_providers=providers,
            )
            selected = await strategy.select_provider(context)
            if selected:
                selections.append(selected.name)

        # Should cycle through all providers
        assert "provider1" in selections
        assert "provider2" in selections
        assert "provider3" in selections


class TestCostBasedStrategy:
    """Tests for CostBasedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_cheapest_provider(self):
        """Should select the cheapest provider."""
        from tests.conftest import MockProvider

        cheap_provider = MockProvider(name="cheap")
        expensive_provider = MockProvider(name="expensive")

        # Override cost estimation
        cheap_provider.estimate_cost = lambda r: CostEstimate(
            prompt_cost=0.001, completion_cost=0.001, total_cost=0.002
        )
        expensive_provider.estimate_cost = lambda r: CostEstimate(
            prompt_cost=0.01, completion_cost=0.01, total_cost=0.02
        )

        strategy = CostBasedStrategy()

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[expensive_provider, cheap_provider],
        )

        selected = await strategy.select_provider(context)
        assert selected is not None
        assert selected.name == "cheap"


class TestLatencyBasedStrategy:
    """Tests for LatencyBasedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_fastest_provider(self):
        """Should select the provider with lowest latency."""
        from tests.conftest import MockProvider

        fast_provider = MockProvider(name="fast", latency_ms=10.0)
        slow_provider = MockProvider(name="slow", latency_ms=100.0)

        strategy = LatencyBasedStrategy()

        context = RoutingContext(
            request=CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="test")],
                request_id="test",
            ),
            available_providers=[slow_provider, fast_provider],
            provider_metrics={
                "fast": ProviderMetrics(name="fast", latency_p50_ms=10.0),
                "slow": ProviderMetrics(name="slow", latency_p50_ms=100.0),
            },
        )

        selected = await strategy.select_provider(context)
        assert selected is not None
        assert selected.name == "fast"


class TestGetStrategy:
    """Tests for get_strategy factory function."""

    def test_get_fallback_strategy(self):
        """Should return FallbackChainStrategy for 'fallback'."""
        strategy = get_strategy("fallback")
        assert isinstance(strategy, FallbackChainStrategy)

    def test_get_round_robin_strategy(self):
        """Should return RoundRobinStrategy for 'round_robin'."""
        strategy = get_strategy("round_robin")
        assert isinstance(strategy, RoundRobinStrategy)

    def test_get_cost_strategy(self):
        """Should return CostBasedStrategy for 'cost'."""
        strategy = get_strategy("cost")
        assert isinstance(strategy, CostBasedStrategy)

    def test_get_latency_strategy(self):
        """Should return LatencyBasedStrategy for 'latency'."""
        strategy = get_strategy("latency")
        assert isinstance(strategy, LatencyBasedStrategy)

    def test_unknown_strategy_raises(self):
        """Should raise ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            get_strategy("unknown")
