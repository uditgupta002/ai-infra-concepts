"""Main router for provider selection.

Coordinates routing strategies, health tracking, and provider filtering
to select the optimal provider for each request.
"""

from llm_gateway.config import GatewaySettings, RoutingStrategy as RoutingStrategyEnum
from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.models import CompletionRequest
from llm_gateway.providers.factory import ProviderFactory
from llm_gateway.routing.strategies import (
    RoutingStrategy,
    RoutingContext,
    FallbackChainStrategy,
    RoundRobinStrategy,
    CostBasedStrategy,
    LatencyBasedStrategy,
    get_strategy,
)
from llm_gateway.routing.health import HealthTracker


class Router:
    """Main router for selecting LLM providers.

    Combines routing strategies with health tracking to make
    intelligent routing decisions.
    """

    def __init__(
        self,
        provider_factory: ProviderFactory,
        health_tracker: HealthTracker,
        default_strategy: RoutingStrategy | None = None,
        settings: GatewaySettings | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            provider_factory: Factory for getting providers.
            health_tracker: Health tracker for provider metrics.
            default_strategy: Default routing strategy.
            settings: Gateway settings.
        """
        self._provider_factory = provider_factory
        self._health_tracker = health_tracker
        self._settings = settings

        # Initialize default strategy
        if default_strategy:
            self._default_strategy = default_strategy
        elif settings:
            self._default_strategy = get_strategy(settings.default_routing_strategy.value)
        else:
            self._default_strategy = FallbackChainStrategy()

        # Cache of strategy instances
        self._strategies: dict[str, RoutingStrategy] = {
            self._default_strategy.name: self._default_strategy,
        }

        # Register providers with health tracker
        for provider in self._provider_factory.get_all().values():
            self._health_tracker.register_provider(provider)

    def get_strategy(self, name: str | None) -> RoutingStrategy:
        """Get a routing strategy by name.

        Args:
            name: Strategy name. If None, returns default.

        Returns:
            RoutingStrategy instance.
        """
        if name is None:
            return self._default_strategy

        if name not in self._strategies:
            self._strategies[name] = get_strategy(name)

        return self._strategies[name]

    async def route(
        self,
        request: CompletionRequest,
    ) -> LLMProvider | None:
        """Route a request to an appropriate provider.

        Args:
            request: The completion request.

        Returns:
            Selected provider or None if no suitable provider found.
        """
        # Get all available providers
        providers = list(self._provider_factory.get_all().values())

        if not providers:
            return None

        # Build routing context
        context = RoutingContext(
            request=request,
            available_providers=providers,
            provider_metrics=self._health_tracker.get_all_metrics(),
            excluded_providers=request.routing.exclude_providers,
            preferred_providers=request.routing.preferred_providers,
        )

        # Handle explicit provider request
        if request.provider:
            provider = self._provider_factory.get(request.provider)
            if provider and self._health_tracker.is_healthy(provider.name):
                return provider
            # Fall through to routing if explicit provider unavailable

        # Get strategy
        strategy_name = request.routing.strategy
        strategy = self.get_strategy(strategy_name)

        # Execute routing
        return await strategy.select_provider(context)

    async def route_with_fallback(
        self,
        request: CompletionRequest,
        failed_providers: list[str] | None = None,
    ) -> LLMProvider | None:
        """Route request, excluding previously failed providers.

        Used for automatic failover when a provider fails.

        Args:
            request: The completion request.
            failed_providers: List of provider names that have failed.

        Returns:
            Selected provider or None.
        """
        # Add failed providers to exclusion list
        if failed_providers:
            # Create a modified routing hints
            exclude = list(request.routing.exclude_providers) + failed_providers
            request.routing.exclude_providers = exclude

        return await self.route(request)

    def get_providers_for_model(self, model: str) -> list[LLMProvider]:
        """Get all providers that support a specific model.

        Args:
            model: Model name.

        Returns:
            List of providers supporting the model.
        """
        result = []
        for provider in self._provider_factory.get_all().values():
            if provider.supports_model(model):
                result.append(provider)
        return result

    def get_healthy_providers(self) -> list[LLMProvider]:
        """Get all currently healthy providers.

        Returns:
            List of healthy providers.
        """
        result = []
        for provider in self._provider_factory.get_all().values():
            if self._health_tracker.is_healthy(provider.name):
                result.append(provider)
        return result

    async def refresh_health(self) -> None:
        """Refresh health status for all providers."""
        await self._health_tracker.check_all_health()
