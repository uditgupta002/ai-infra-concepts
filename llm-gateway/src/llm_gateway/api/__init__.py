"""API layer for LLM Gateway."""

from llm_gateway.api.routes import router
from llm_gateway.api.schemas import (
    CompletionRequestSchema,
    CompletionResponseSchema,
    ProviderListResponse,
    HealthResponse,
    CacheStatsResponse,
)
from llm_gateway.api.middleware import (
    RequestLoggingMiddleware,
    CorrelationIdMiddleware,
    ErrorHandlingMiddleware,
)

__all__ = [
    "router",
    "CompletionRequestSchema",
    "CompletionResponseSchema",
    "ProviderListResponse",
    "HealthResponse",
    "CacheStatsResponse",
    "RequestLoggingMiddleware",
    "CorrelationIdMiddleware",
    "ErrorHandlingMiddleware",
]
