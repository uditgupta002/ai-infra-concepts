"""Test configuration for LLM Gateway."""

import pytest
import asyncio
from typing import AsyncIterator

from llm_gateway.providers.base import LLMProvider, ProviderCapability
from llm_gateway.providers.models import (
    Message,
    MessageRole,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    TokenUsage,
    ProviderHealth,
    ProviderInfo,
    CostEstimate,
)


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        should_fail: bool = False,
        latency_ms: float = 10.0,
    ) -> None:
        self._name = name
        self._should_fail = should_fail
        self._latency_ms = latency_ms
        self._call_count = 0
        self._supported_models = ["mock-model", "mock-model-2"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"Mock Provider ({self._name})"

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def supported_models(self) -> list[str]:
        return self._supported_models

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1

        if self._should_fail:
            from llm_gateway.providers.base import ProviderError
            raise ProviderError(
                message="Mock failure",
                provider=self._name,
                status_code=500,
                retryable=True,
            )

        import asyncio
        await asyncio.sleep(self._latency_ms / 1000)

        return CompletionResponse(
            id=f"mock-{self._call_count}",
            content="Mock response",
            model=request.model or self.default_model,
            provider=self._name,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            request_id=request.request_id,
        )

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        self._call_count += 1

        if self._should_fail:
            from llm_gateway.providers.base import ProviderError
            raise ProviderError(
                message="Mock failure",
                provider=self._name,
                status_code=500,
                retryable=True,
            )

        chunks = ["Mock ", "streaming ", "response"]
        for i, text in enumerate(chunks):
            yield StreamChunk(
                id=f"chunk-{i}",
                content=text,
                model=request.model or self.default_model,
                provider=self._name,
            )

        yield StreamChunk(
            id="chunk-final",
            content="",
            model=request.model or self.default_model,
            provider=self._name,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            is_healthy=not self._should_fail,
            latency_ms=self._latency_ms,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def estimate_cost(self, request: CompletionRequest) -> CostEstimate:
        return CostEstimate(
            prompt_cost=0.001,
            completion_cost=0.002,
            total_cost=0.003,
            currency="USD",
        )

    def supports_model(self, model: str) -> bool:
        return model in self._supported_models


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock provider."""
    return MockProvider()


@pytest.fixture
def failing_provider() -> MockProvider:
    """Create a failing mock provider."""
    return MockProvider(name="failing", should_fail=True)


@pytest.fixture
def sample_request() -> CompletionRequest:
    """Create a sample completion request."""
    return CompletionRequest(
        messages=[
            Message(role=MessageRole.USER, content="Hello, world!")
        ],
        temperature=0.7,
        max_tokens=100,
        request_id="test-request-1",
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello!"),
    ]


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
