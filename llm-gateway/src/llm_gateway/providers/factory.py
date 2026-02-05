"""Provider factory for creating and managing LLM provider instances.

Implements the factory pattern for lazy instantiation and caching
of provider instances.
"""

from functools import lru_cache
from typing import Callable

from llm_gateway.config import GatewaySettings, get_gateway_settings
from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.openai_provider import OpenAIProvider
from llm_gateway.providers.anthropic_provider import AnthropicProvider


# Type for provider factory functions
ProviderFactoryFunc = Callable[[GatewaySettings], LLMProvider | None]


class ProviderFactory:
    """Factory for creating and managing LLM provider instances.

    Handles lazy instantiation, caching, and registration of providers.
    Providers are only instantiated when first requested and if their
    API keys are configured.
    """

    def __init__(self, settings: GatewaySettings | None = None) -> None:
        """Initialize the provider factory.

        Args:
            settings: Gateway settings. If None, loads from environment.
        """
        self._settings = settings or get_gateway_settings()
        self._providers: dict[str, LLMProvider] = {}
        self._factory_funcs: dict[str, ProviderFactoryFunc] = {}

        # Register built-in providers
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        """Register the built-in provider factories."""
        self.register("openai", self._create_openai_provider)
        self.register("anthropic", self._create_anthropic_provider)

    def _create_openai_provider(
        self,
        settings: GatewaySettings,
    ) -> OpenAIProvider | None:
        """Create an OpenAI provider instance.

        Args:
            settings: Gateway settings.

        Returns:
            OpenAIProvider instance or None if not configured.
        """
        if not settings.openai_api_key:
            return None

        return OpenAIProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            default_model=settings.openai_default_model,
            timeout=settings.request_timeout,
            connect_timeout=settings.connect_timeout,
        )

    def _create_anthropic_provider(
        self,
        settings: GatewaySettings,
    ) -> AnthropicProvider | None:
        """Create an Anthropic provider instance.

        Args:
            settings: Gateway settings.

        Returns:
            AnthropicProvider instance or None if not configured.
        """
        if not settings.anthropic_api_key:
            return None

        return AnthropicProvider(
            api_key=settings.anthropic_api_key.get_secret_value(),
            base_url=settings.anthropic_base_url,
            default_model=settings.anthropic_default_model,
            timeout=settings.request_timeout,
            connect_timeout=settings.connect_timeout,
        )

    def register(
        self,
        name: str,
        factory_func: ProviderFactoryFunc,
    ) -> None:
        """Register a provider factory function.

        Args:
            name: Provider name.
            factory_func: Function that creates the provider instance.
        """
        self._factory_funcs[name] = factory_func

    def get(self, name: str) -> LLMProvider | None:
        """Get a provider instance by name.

        Providers are lazily instantiated and cached.

        Args:
            name: Provider name (e.g., 'openai', 'anthropic').

        Returns:
            LLMProvider instance or None if not available.
        """
        # Return cached instance if available
        if name in self._providers:
            return self._providers[name]

        # Try to create the provider
        factory_func = self._factory_funcs.get(name)
        if factory_func is None:
            return None

        provider = factory_func(self._settings)
        if provider is not None:
            self._providers[name] = provider

        return provider

    def get_all(self) -> dict[str, LLMProvider]:
        """Get all available providers.

        Instantiates any providers that haven't been created yet.

        Returns:
            Dictionary mapping provider names to instances.
        """
        for name in self._factory_funcs:
            self.get(name)
        return self._providers.copy()

    def get_available(self) -> list[str]:
        """Get list of available provider names.

        A provider is available if it is configured with valid credentials.

        Returns:
            List of available provider names.
        """
        available = []
        for name in self._factory_funcs:
            if self.get(name) is not None:
                available.append(name)
        return available

    def is_available(self, name: str) -> bool:
        """Check if a provider is available.

        Args:
            name: Provider name.

        Returns:
            True if the provider is available.
        """
        return self.get(name) is not None

    def supports_model(self, model: str) -> list[str]:
        """Get providers that support a specific model.

        Args:
            model: Model name to check.

        Returns:
            List of provider names that support the model.
        """
        providers = []
        for name, provider in self.get_all().items():
            if provider.supports_model(model):
                providers.append(name)
        return providers


# Cached factory instance
_factory: ProviderFactory | None = None


def get_provider_factory(
    settings: GatewaySettings | None = None,
) -> ProviderFactory:
    """Get the cached provider factory instance.

    Args:
        settings: Optional settings override. If provided,
                  creates a new factory instance.

    Returns:
        ProviderFactory instance.
    """
    global _factory

    if settings is not None:
        # Create new factory with custom settings
        return ProviderFactory(settings)

    if _factory is None:
        _factory = ProviderFactory()

    return _factory


def reset_provider_factory() -> None:
    """Reset the cached provider factory.

    Useful for testing or when settings change.
    """
    global _factory
    _factory = None
