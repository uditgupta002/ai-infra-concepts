"""OpenAI provider implementation.

Provides integration with OpenAI's chat completion API including
GPT-4, GPT-4-Turbo, and GPT-3.5-Turbo models.
"""

import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
import tiktoken

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
)


# Model pricing per 1000 tokens (USD)
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Model to tiktoken encoding mapping
MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
}


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation.

    Handles communication with OpenAI's API for chat completions,
    including streaming and token counting.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4-turbo-preview",
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
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
        self._encoders: dict[str, tiktoken.Encoding] = {}

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    @property
    def supported_models(self) -> list[str]:
        return list(OPENAI_PRICING.keys())

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.VISION,
            ProviderCapability.JSON_MODE,
        ]

    def _get_client(self) -> httpx.AsyncClient:
        """Create an HTTP client for API calls."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                timeout=self._timeout,
                connect=self._connect_timeout,
            ),
        )

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get the tiktoken encoder for a model.

        Args:
            model: Model name.

        Returns:
            Tiktoken encoder.
        """
        encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens in.
            model: Model to use for encoding.

        Returns:
            Number of tokens.
        """
        model = model or self._default_model
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))

    def _count_messages_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> int:
        """Count tokens in a list of messages.

        Args:
            messages: List of message dictionaries.
            model: Model name for encoding.

        Returns:
            Total token count.
        """
        encoder = self._get_encoder(model)
        tokens_per_message = 3  # <|start|>{role}\n{content}<|end|>\n
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

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
        pricing = OPENAI_PRICING.get(model, {"input": 0.01, "output": 0.03})

        # Count input tokens
        messages = [msg.to_openai_format() for msg in request.messages]
        input_tokens = self._count_messages_tokens(messages, model)

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
        """Generate a completion using OpenAI's API.

        Args:
            request: The completion request.

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ProviderError: If the request fails.
        """
        model = self.get_model(request)
        start_time = time.perf_counter()

        payload: dict[str, Any] = {
            "model": model,
            "messages": [msg.to_openai_format() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self._get_client() as client:
            try:
                response = await client.post("/chat/completions", json=payload)

                if response.status_code != 200:
                    self._handle_error(response)

                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000

                usage = TokenUsage(
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    completion_tokens=data["usage"]["completion_tokens"],
                    total_tokens=data["usage"]["total_tokens"],
                )

                pricing = OPENAI_PRICING.get(model, {"input": 0.01, "output": 0.03})
                cost = CostEstimate.from_usage(
                    usage=usage,
                    input_price_per_1k=pricing["input"],
                    output_price_per_1k=pricing["output"],
                )

                return CompletionResponse(
                    id=data["id"],
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    provider=self.name,
                    usage=usage,
                    cost=cost,
                    latency_ms=latency_ms,
                    request_id=request.request_id,
                    finish_reason=data["choices"][0].get("finish_reason"),
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": [msg.to_openai_format() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._handle_error(response)

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            import json

                            chunk_data = json.loads(data)
                            choices = chunk_data.get("choices", [])

                            if not choices:
                                # Usage info comes in final message
                                usage_data = chunk_data.get("usage")
                                if usage_data:
                                    yield StreamChunk(
                                        id=chunk_id,
                                        content="",
                                        model=model,
                                        provider=self.name,
                                        finish_reason="stop",
                                        usage=TokenUsage(
                                            prompt_tokens=usage_data["prompt_tokens"],
                                            completion_tokens=usage_data[
                                                "completion_tokens"
                                            ],
                                            total_tokens=usage_data["total_tokens"],
                                        ),
                                    )
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = choice.get("finish_reason")

                            if content or finish_reason:
                                yield StreamChunk(
                                    id=chunk_id,
                                    content=content or "",
                                    model=model,
                                    provider=self.name,
                                    finish_reason=finish_reason,
                                )

                        except Exception:
                            continue  # Skip malformed chunks

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
        """Check the health of the OpenAI API.

        Returns:
            ProviderHealth status.
        """
        start_time = time.perf_counter()

        try:
            async with self._get_client() as client:
                # Use a minimal request to check connectivity
                response = await client.get("/models")
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
        """Get pricing information for OpenAI models."""
        return OPENAI_PRICING
