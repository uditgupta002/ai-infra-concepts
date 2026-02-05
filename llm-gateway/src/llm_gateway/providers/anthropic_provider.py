"""Anthropic provider implementation.

Provides integration with Anthropic's Claude API including
Claude 3.5 and Claude 3 family models.
"""

import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from llm_gateway.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ServiceUnavailableError,
)
from llm_gateway.providers.models import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ProviderHealth,
    TokenUsage,
    CostEstimate,
    MessageRole,
)


# Model pricing per 1000 tokens (USD)
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}

# API version
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation.

    Handles communication with Anthropic's API for Claude models,
    including streaming and token counting.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-3-sonnet-20240229",
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key.
            base_url: API base URL.
            default_model: Default model to use.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._connect_timeout = connect_timeout

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic"

    @property
    def supported_models(self) -> list[str]:
        return list(ANTHROPIC_PRICING.keys())

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
        ]

    def _get_client(self) -> httpx.AsyncClient:
        """Create an HTTP client for API calls."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                timeout=self._timeout,
                connect=self._connect_timeout,
            ),
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate token count for text.

        Anthropic doesn't provide a public tokenizer, so we approximate
        using a simple heuristic (roughly 4 characters per token for English).

        Args:
            text: Text to count tokens in.
            model: Model (unused, kept for interface compatibility).

        Returns:
            Estimated number of tokens.
        """
        # Rough approximation: ~4 characters per token for English text
        return len(text) // 4

    def _format_messages(
        self,
        messages: list[Any],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Format messages for Anthropic's API.

        Anthropic requires system messages to be passed separately from
        the conversation messages.

        Args:
            messages: List of Message objects.

        Returns:
            Tuple of (system_prompt, formatted_messages).
        """
        system_prompt = None
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                # Anthropic uses 'user' and 'assistant' roles
                role = "user" if msg.role == MessageRole.USER else "assistant"
                formatted.append({"role": role, "content": msg.content})

        # Ensure messages alternate between user and assistant
        # Anthropic requires the conversation to start with a user message
        if formatted and formatted[0]["role"] != "user":
            # Prepend an empty user message if needed
            formatted.insert(0, {"role": "user", "content": ""})

        return system_prompt, formatted

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
        model = self.get_model(request)
        pricing = ANTHROPIC_PRICING.get(model, {"input": 0.003, "output": 0.015})

        # Count input tokens (approximation)
        input_text = " ".join(msg.content for msg in request.messages)
        input_tokens = self.count_tokens(input_text)

        # Estimate output tokens
        output_tokens = estimated_output_tokens or request.max_tokens or 500

        usage = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return CostEstimate.from_usage(
            usage=usage,
            input_price_per_1k=pricing["input"],
            output_price_per_1k=pricing["output"],
        )

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses.

        Args:
            response: HTTP response.

        Raises:
            Appropriate ProviderError subclass.
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        status = response.status_code

        if status == 401:
            raise AuthenticationError(error_message, self.name)
        elif status == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                error_message,
                self.name,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif status == 400:
            raise InvalidRequestError(error_message, self.name)
        elif status >= 500:
            raise ServiceUnavailableError(error_message, self.name)
        else:
            raise ProviderError(
                message=error_message,
                provider=self.name,
                status_code=status,
                retryable=status >= 500,
            )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Anthropic's API.

        Args:
            request: The completion request.

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ProviderError: If the request fails.
        """
        model = self.get_model(request)
        start_time = time.perf_counter()

        system_prompt, messages = self._format_messages(request.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if request.temperature != 0.7:  # Only include if non-default
            payload["temperature"] = request.temperature

        if request.top_p != 1.0:
            payload["top_p"] = request.top_p

        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        async with self._get_client() as client:
            try:
                response = await client.post("/v1/messages", json=payload)

                if response.status_code != 200:
                    self._handle_error(response)

                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Extract content from response
                content_blocks = data.get("content", [])
                content = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        content += block.get("text", "")

                usage = TokenUsage(
                    prompt_tokens=data["usage"]["input_tokens"],
                    completion_tokens=data["usage"]["output_tokens"],
                    total_tokens=(
                        data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                    ),
                )

                pricing = ANTHROPIC_PRICING.get(model, {"input": 0.003, "output": 0.015})
                cost = CostEstimate.from_usage(
                    usage=usage,
                    input_price_per_1k=pricing["input"],
                    output_price_per_1k=pricing["output"],
                )

                return CompletionResponse(
                    id=data["id"],
                    content=content,
                    model=data["model"],
                    provider=self.name,
                    usage=usage,
                    cost=cost,
                    latency_ms=latency_ms,
                    request_id=request.request_id,
                    finish_reason=data.get("stop_reason"),
                )

            except httpx.TimeoutException as e:
                raise ProviderError(
                    message=f"Request timed out: {e}",
                    provider=self.name,
                    retryable=True,
                    original_error=e,
                )
            except httpx.RequestError as e:
                raise ProviderError(
                    message=f"Request failed: {e}",
                    provider=self.name,
                    retryable=True,
                    original_error=e,
                )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            request: The completion request.

        Yields:
            StreamChunk objects as they become available.

        Raises:
            ProviderError: If the request fails.
        """
        model = self.get_model(request)
        chunk_id = str(uuid4())

        system_prompt, messages = self._format_messages(request.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if request.temperature != 0.7:
            payload["temperature"] = request.temperature

        if request.top_p != 1.0:
            payload["top_p"] = request.top_p

        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "POST",
                    "/v1/messages",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._handle_error(response)

                    input_tokens = 0
                    output_tokens = 0

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:]  # Remove "data: " prefix

                        try:
                            import json

                            event_data = json.loads(data)
                            event_type = event_data.get("type")

                            if event_type == "message_start":
                                # Get usage from message start
                                usage = event_data.get("message", {}).get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)

                            elif event_type == "content_block_delta":
                                delta = event_data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield StreamChunk(
                                            id=chunk_id,
                                            content=text,
                                            model=model,
                                            provider=self.name,
                                        )

                            elif event_type == "message_delta":
                                # Final message with stop reason and usage
                                delta = event_data.get("delta", {})
                                stop_reason = delta.get("stop_reason")
                                usage = event_data.get("usage", {})
                                output_tokens = usage.get("output_tokens", 0)

                                yield StreamChunk(
                                    id=chunk_id,
                                    content="",
                                    model=model,
                                    provider=self.name,
                                    finish_reason=stop_reason,
                                    usage=TokenUsage(
                                        prompt_tokens=input_tokens,
                                        completion_tokens=output_tokens,
                                        total_tokens=input_tokens + output_tokens,
                                    ),
                                )

                        except Exception:
                            continue  # Skip malformed events

            except httpx.TimeoutException as e:
                raise ProviderError(
                    message=f"Stream timed out: {e}",
                    provider=self.name,
                    retryable=True,
                    original_error=e,
                )
            except httpx.RequestError as e:
                raise ProviderError(
                    message=f"Stream failed: {e}",
                    provider=self.name,
                    retryable=True,
                    original_error=e,
                )

    async def health_check(self) -> ProviderHealth:
        """Check the health of the Anthropic API.

        Returns:
            ProviderHealth status.
        """
        start_time = time.perf_counter()

        try:
            async with self._get_client() as client:
                # Anthropic doesn't have a dedicated health endpoint
                # Use a minimal message request to verify connectivity
                payload = {
                    "model": self._default_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}],
                }
                response = await client.post("/v1/messages", json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000

                is_healthy = response.status_code == 200

                return ProviderHealth(
                    is_healthy=is_healthy,
                    latency_ms=latency_ms,
                    success_rate=1.0 if is_healthy else 0.0,
                    last_check=datetime.now(timezone.utc),
                )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ProviderHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                success_rate=0.0,
                error_count=1,
                last_error=str(e),
                last_check=datetime.now(timezone.utc),
            )

    def _get_pricing(self) -> dict[str, dict[str, float]]:
        """Get pricing information for Anthropic models."""
        return ANTHROPIC_PRICING
