"""Shared models for AI Infrastructure modules.

Provides base classes and common models used across all modules.
"""

from shared.models.base import (
    BaseRequest,
    BaseResponse,
    ErrorResponse,
    ErrorDetail,
    PaginatedResponse,
    HealthStatus,
    HealthCheck,
)

__all__ = [
    "BaseRequest",
    "BaseResponse",
    "ErrorResponse",
    "ErrorDetail",
    "PaginatedResponse",
    "HealthStatus",
    "HealthCheck",
]
