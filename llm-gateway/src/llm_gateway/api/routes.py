"""API route definitions.

Defines FastAPI routes for the LLM Gateway API.
"""

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_gateway import __version__
from llm_gateway.api.schemas import (
    CompletionRequestSchema,
    CompletionResponseSchema,
    ProviderListResponse,
    ProviderInfoSchema,
    HealthResponse,
    CacheStatsResponse,
    ErrorResponse,
)
from llm_gateway.providers.models import (
    CompletionRequest,
    Message,
    ProviderHealth,
)
from llm_gateway.providers.base import ProviderError
from llm_gateway.resilience.rate_limiter import RateLimitExceeded
from llm_gateway.resilience.circuit_breaker import CircuitOpenError


logger = logging.getLogger(__name__)

router = APIRouter()


def get_gateway(request: Request):
    """Dependency to get the gateway instance from app state."""
    return request.app.state.gateway


def get_rate_limiter(request: Request):
    """Dependency to get the rate limiter from app state."""
    return request.app.state.rate_limiter


@router.post(
    "/v1/completions",
    response_model=CompletionResponseSchema,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": ErrorResponse, "description": "Provider error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Create a completion",
    description="Generate a chat completion using the configured LLM providers.",
)
async def create_completion(
    body: CompletionRequestSchema,
    request: Request,
    gateway=Depends(get_gateway),
    rate_limiter=Depends(get_rate_limiter),
) -> CompletionResponseSchema:
    """Create a chat completion.

    Handles provider selection, caching, rate limiting, and resilience.
    """
    # Get rate limit key
    rate_key = body.user_id or request.client.host if request.client else "anonymous"

    # Check rate limit
    if rate_limiter:
        try:
            await rate_limiter.acquire(rate_key)
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": str(e),
                    "retry_after": e.retry_after,
                },
                headers={"Retry-After": str(int(e.retry_after or 60))},
            )

    # Build completion request
    completion_request = CompletionRequest(
        messages=body.messages,
        model=body.model,
        provider=body.provider,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        stop_sequences=body.stop_sequences,
        routing=body.routing,
        request_id=str(uuid4()),
        user_id=body.user_id,
        metadata=body.metadata,
        skip_cache=body.skip_cache,
    )

    try:
        response = await gateway.complete(completion_request)
        return CompletionResponseSchema(**response.model_dump())

    except CircuitOpenError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "circuit_open",
                "message": f"Provider circuit is open: {e.name}",
                "retry_after": e.time_until_retry,
            },
        )
    except ProviderError as e:
        status_code = e.status_code or 502
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": "provider_error",
                "message": str(e),
                "provider": e.provider,
                "retryable": e.retryable,
            },
        )
    except Exception as e:
        logger.exception("Unexpected error in completion")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred",
            },
        )


@router.post(
    "/v1/completions/stream",
    summary="Create a streaming completion",
    description="Generate a streaming chat completion using Server-Sent Events.",
)
async def create_completion_stream(
    body: CompletionRequestSchema,
    request: Request,
    gateway=Depends(get_gateway),
    rate_limiter=Depends(get_rate_limiter),
) -> StreamingResponse:
    """Create a streaming chat completion.

    Returns Server-Sent Events with completion chunks.
    """
    # Get rate limit key
    rate_key = body.user_id or request.client.host if request.client else "anonymous"

    # Check rate limit
    if rate_limiter:
        try:
            await rate_limiter.acquire(rate_key)
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": str(e),
                    "retry_after": e.retry_after,
                },
            )

    # Build completion request
    completion_request = CompletionRequest(
        messages=body.messages,
        model=body.model,
        provider=body.provider,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        stop_sequences=body.stop_sequences,
        routing=body.routing,
        request_id=str(uuid4()),
        user_id=body.user_id,
        metadata=body.metadata,
        skip_cache=True,  # Streaming doesn't use cache
    )

    async def generate_stream() -> AsyncIterator[str]:
        """Generate SSE stream."""
        try:
            async for chunk in gateway.stream(completion_request):
                data = chunk.model_dump()
                yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/v1/providers",
    response_model=ProviderListResponse,
    summary="List providers",
    description="Get a list of all available LLM providers.",
)
async def list_providers(
    gateway=Depends(get_gateway),
) -> ProviderListResponse:
    """List all available providers."""
    providers = await gateway.get_providers()

    provider_schemas = [
        ProviderInfoSchema(
            name=p.name,
            display_name=p.display_name,
            is_available=p.is_available,
            supported_models=p.supported_models,
            default_model=p.default_model,
            capabilities=p.capabilities,
            health=p.health,
        )
        for p in providers
    ]

    return ProviderListResponse(
        providers=provider_schemas,
        total=len(provider_schemas),
    )


@router.get(
    "/v1/providers/{provider_id}/health",
    response_model=ProviderHealth,
    summary="Get provider health",
    description="Get health status for a specific provider.",
)
async def get_provider_health(
    provider_id: str,
    gateway=Depends(get_gateway),
) -> ProviderHealth:
    """Get health status for a provider."""
    health = await gateway.get_provider_health(provider_id)

    if health is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"Provider not found: {provider_id}"},
        )

    return health


@router.get(
    "/v1/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get cache statistics",
    description="Get statistics about the semantic cache.",
)
async def get_cache_stats(
    gateway=Depends(get_gateway),
) -> CacheStatsResponse:
    """Get cache statistics."""
    stats = await gateway.get_cache_stats()
    return CacheStatsResponse(
        enabled=stats.get("enabled", False),
        size=stats.get("size", 0),
        hits=stats.get("hits", 0),
        misses=stats.get("misses", 0),
        hit_rate=stats.get("hit_rate", 0.0),
        evictions=stats.get("evictions", 0),
    )


@router.delete(
    "/v1/cache",
    summary="Clear cache",
    description="Clear all cached responses.",
)
async def clear_cache(
    gateway=Depends(get_gateway),
) -> dict:
    """Clear the cache."""
    await gateway.clear_cache()
    return {"status": "ok", "message": "Cache cleared"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the gateway and its providers.",
)
async def health_check(
    gateway=Depends(get_gateway),
) -> HealthResponse:
    """Perform a health check."""
    provider_health = await gateway.check_health()

    # Determine overall status
    unhealthy_count = sum(1 for h in provider_health.values() if not h.is_healthy)
    if unhealthy_count == len(provider_health):
        status = "unhealthy"
    elif unhealthy_count > 0:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        providers=provider_health,
        cache_enabled=gateway.cache_enabled,
        rate_limiting_enabled=gateway.rate_limiting_enabled,
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="Basic information about the API.",
)
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "LLM Gateway",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
