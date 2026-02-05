"""Routing strategies for provider selection.

Implements various strategies for selecting which LLM provider
to route a request to based on different criteria.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
from typing import Any

from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.models import CompletionRequest, ProviderHealth


@dataclass
class ProviderMetrics:
    """Metrics for a provider used in routing decisions."""

    name: str
    is_healthy: bool = True
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    request_count: int = 0
    estimated_cost_per_1k: float = 0.0
    last_used: datetime | None = None


@dataclass
class RoutingContext:
    """Context for making routing decisions."""

    request: CompletionRequest
    available_providers: list[LLMProvider]
    provider_metrics: dict[str, ProviderMetrics] = field(default_factory=dict)
    excluded_providers: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies.

    Routing strategies determine which provider should handle a given
    request based on various criteria (cost, latency, availability, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        ...

    @abstractmethod
    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select a provider for the given request.

        Args:
            context: Routing context with request and provider info.

        Returns:
            Selected provider or None if no suitable provider found.
        """
        ...

    def filter_providers(
        self,
        context: RoutingContext,
    ) -> list[LLMProvider]:
        """Filter providers based on context constraints.

        Applies common filtering logic:
        - Exclude providers in the exclude list
        - Filter to only healthy providers
        - Filter to providers that support the requested model

        Args:
            context: Routing context.

        Returns:
            Filtered list of providers.
        """
        providers = []

        for provider in context.available_providers:
            # Skip excluded providers
            if provider.name in context.excluded_providers:
                continue

            # Skip unhealthy providers
            metrics = context.provider_metrics.get(provider.name)
            if metrics and not metrics.is_healthy:
                continue

            # Check model support if specific model requested
            if context.request.model:
                if not provider.supports_model(context.request.model):
                    continue

            providers.append(provider)

        return providers

    def prioritize_preferred(
        self,
        providers: list[LLMProvider],
        preferred: list[str],
    ) -> list[LLMProvider]:
        """Reorder providers to prioritize preferred ones.

        Args:
            providers: List of providers.
            preferred: List of preferred provider names.

        Returns:
            Reordered list with preferred providers first.
        """
        if not preferred:
            return providers

        preferred_set = set(preferred)
        preferred_providers = [p for p in providers if p.name in preferred_set]
        other_providers = [p for p in providers if p.name not in preferred_set]

        # Sort preferred by their order in the preference list
        preferred_providers.sort(key=lambda p: preferred.index(p.name))

        return preferred_providers + other_providers


class FallbackChainStrategy(RoutingStrategy):
    """Fallback chain routing strategy.

    Tries providers in a defined priority order, falling back to the
    next provider if the current one is unavailable.
    """

    def __init__(self, priority_order: list[str] | None = None) -> None:
        """Initialize fallback chain strategy.

        Args:
            priority_order: List of provider names in priority order.
                          If None, uses default order.
        """
        self._priority_order = priority_order or ["openai", "anthropic"]

    @property
    def name(self) -> str:
        return "fallback"

    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select first available provider in priority order.

        Args:
            context: Routing context.

        Returns:
            First available provider or None.
        """
        providers = self.filter_providers(context)

        if not providers:
            return None

        # Apply user preferences if specified
        if context.preferred_providers:
            providers = self.prioritize_preferred(
                providers,
                context.preferred_providers,
            )
            if providers:
                return providers[0]

        # Use default priority order
        provider_map = {p.name: p for p in providers}

        for name in self._priority_order:
            if name in provider_map:
                return provider_map[name]

        # Return first available if none match priority
        return providers[0] if providers else None


class RoundRobinStrategy(RoutingStrategy):
    """Round-robin routing strategy.

    Distributes requests evenly across available providers.
    """

    def __init__(self) -> None:
        """Initialize round-robin strategy."""
        self._counters: dict[str, int] = defaultdict(int)
        self._index = 0

    @property
    def name(self) -> str:
        return "round_robin"

    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select next provider in round-robin order.

        Args:
            context: Routing context.

        Returns:
            Next provider in rotation.
        """
        providers = self.filter_providers(context)

        if not providers:
            return None

        # Apply preferences if specified
        if context.preferred_providers:
            providers = self.prioritize_preferred(
                providers,
                context.preferred_providers,
            )

        # Get next provider in rotation
        self._index = (self._index + 1) % len(providers)
        return providers[self._index]


class CostBasedStrategy(RoutingStrategy):
    """Cost-based routing strategy.

    Selects the cheapest provider that can handle the request.
    """

    @property
    def name(self) -> str:
        return "cost"

    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select cheapest available provider.

        Args:
            context: Routing context.

        Returns:
            Cheapest provider or None.
        """
        providers = self.filter_providers(context)

        if not providers:
            return None

        # Estimate cost for each provider
        provider_costs: list[tuple[LLMProvider, float]] = []

        for provider in providers:
            try:
                estimate = provider.estimate_cost(context.request)
                provider_costs.append((provider, estimate.total_cost))
            except Exception:
                # If cost estimation fails, use a high default
                provider_costs.append((provider, float("inf")))

        # Sort by cost (ascending)
        provider_costs.sort(key=lambda x: x[1])

        # Apply preference as tie-breaker
        if context.preferred_providers:
            # Among providers with similar cost (within 10%), prefer the preferred one
            if len(provider_costs) >= 2:
                cheapest_cost = provider_costs[0][1]
                similar_cost_providers = [
                    p for p, c in provider_costs if c <= cheapest_cost * 1.1
                ]

                for preferred_name in context.preferred_providers:
                    for provider in similar_cost_providers:
                        if provider.name == preferred_name:
                            return provider

        return provider_costs[0][0] if provider_costs else None


class LatencyBasedStrategy(RoutingStrategy):
    """Latency-based routing strategy.

    Selects the provider with the lowest latency.
    """

    @property
    def name(self) -> str:
        return "latency"

    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select provider with lowest latency.

        Args:
            context: Routing context.

        Returns:
            Fastest provider or None.
        """
        providers = self.filter_providers(context)

        if not providers:
            return None

        # Sort by latency
        provider_latencies: list[tuple[LLMProvider, float]] = []

        for provider in providers:
            metrics = context.provider_metrics.get(provider.name)
            latency = metrics.latency_p50_ms if metrics else float("inf")
            provider_latencies.append((provider, latency))

        # Sort by latency (ascending)
        provider_latencies.sort(key=lambda x: x[1])

        # Apply preference as tie-breaker for similar latencies
        if context.preferred_providers:
            if len(provider_latencies) >= 2:
                fastest_latency = provider_latencies[0][1]
                # Consider latencies within 20% as similar
                similar_latency_providers = [
                    p for p, l in provider_latencies if l <= fastest_latency * 1.2
                ]

                for preferred_name in context.preferred_providers:
                    for provider in similar_latency_providers:
                        if provider.name == preferred_name:
                            return provider

        return provider_latencies[0][0] if provider_latencies else None


class WeightedRandomStrategy(RoutingStrategy):
    """Weighted random routing strategy.

    Selects providers randomly with weights based on historical performance.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize weighted random strategy.

        Args:
            weights: Optional fixed weights. If None, weights are
                    calculated from metrics.
        """
        self._fixed_weights = weights

    @property
    def name(self) -> str:
        return "weighted_random"

    async def select_provider(
        self,
        context: RoutingContext,
    ) -> LLMProvider | None:
        """Select provider randomly based on weights.

        Args:
            context: Routing context.

        Returns:
            Randomly selected provider.
        """
        providers = self.filter_providers(context)

        if not providers:
            return None

        if len(providers) == 1:
            return providers[0]

        # Calculate weights
        weights = []
        for provider in providers:
            if self._fixed_weights and provider.name in self._fixed_weights:
                weights.append(self._fixed_weights[provider.name])
            else:
                # Calculate weight from metrics (higher success rate = higher weight)
                metrics = context.provider_metrics.get(provider.name)
                if metrics:
                    # Weight based on success rate and inverse of latency
                    weight = metrics.success_rate
                    if metrics.latency_p50_ms > 0:
                        weight *= 1000 / metrics.latency_p50_ms
                    weights.append(max(weight, 0.1))
                else:
                    weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(providers)] * len(providers)

        # Random selection based on weights
        r = random.random()
        cumulative = 0.0
        for provider, weight in zip(providers, weights):
            cumulative += weight
            if r <= cumulative:
                return provider

        return providers[-1]


def get_strategy(name: str) -> RoutingStrategy:
    """Get a routing strategy by name.

    Args:
        name: Strategy name.

    Returns:
        RoutingStrategy instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    strategies: dict[str, type[RoutingStrategy]] = {
        "fallback": FallbackChainStrategy,
        "round_robin": RoundRobinStrategy,
        "cost": CostBasedStrategy,
        "latency": LatencyBasedStrategy,
        "weighted_random": WeightedRandomStrategy,
    }

    strategy_class = strategies.get(name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown routing strategy: {name}")

    return strategy_class()
