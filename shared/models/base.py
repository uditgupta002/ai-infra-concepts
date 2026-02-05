"""Base models and common types for AI Infrastructure.

This module provides foundational Pydantic models that serve as base classes
for request/response patterns across all modules.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


T = TypeVar("T")


class BaseRequest(BaseModel):
    """Base class for all API requests.

    Provides common fields and configuration for request validation.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        str_strip_whitespace=True,  # Strip whitespace from strings
        validate_default=True,  # Validate default values
    )

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this request",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata to attach to the request",
    )


class BaseResponse(BaseModel):
    """Base class for all API responses.

    Provides common fields for response tracking and metadata.
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore extra fields in responses
    )

    request_id: str = Field(
        description="Request ID this response corresponds to",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp in UTC",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class ErrorDetail(BaseModel):
    """Detailed error information for debugging."""

    code: str = Field(description="Error code for programmatic handling")
    message: str = Field(description="Human-readable error message")
    field: str | None = Field(
        default=None,
        description="Field that caused the error, if applicable",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context",
    )


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(description="Request ID for correlation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp in UTC",
    )
    error: ErrorDetail = Field(description="Error details")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="Additional errors if multiple occurred",
    )

    @classmethod
    def from_exception(
        cls,
        request_id: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ErrorResponse":
        """Create an error response from an exception.

        Args:
            request_id: The request ID to correlate with.
            code: Error code for programmatic handling.
            message: Human-readable error message.
            details: Optional additional error context.

        Returns:
            ErrorResponse instance.
        """
        return cls(
            request_id=request_id,
            error=ErrorDetail(
                code=code,
                message=message,
                details=details or {},
            ),
        )


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    model_config = ConfigDict(extra="ignore")

    items: list[T] = Field(description="List of items in this page")
    total: int = Field(description="Total number of items across all pages")
    page: int = Field(default=1, description="Current page number (1-indexed)")
    page_size: int = Field(default=20, description="Number of items per page")
    has_next: bool = Field(description="Whether there are more pages")
    has_previous: bool = Field(description="Whether there are previous pages")

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int = 1,
        page_size: int = 20,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response.

        Args:
            items: Items for the current page.
            total: Total number of items.
            page: Current page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            PaginatedResponse instance.
        """
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total,
            has_previous=page > 1,
        )


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """Health check response model."""

    model_config = ConfigDict(extra="ignore")

    status: HealthStatus = Field(description="Overall health status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )
    checks: dict[str, HealthStatus] = Field(
        default_factory=dict,
        description="Individual component health checks",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health check details",
    )

    @classmethod
    def healthy(cls, version: str, **details: Any) -> "HealthCheck":
        """Create a healthy status response.

        Args:
            version: Application version string.
            **details: Optional additional details.

        Returns:
            HealthCheck with healthy status.
        """
        return cls(
            status=HealthStatus.HEALTHY,
            version=version,
            details=details,
        )

    @classmethod
    def degraded(
        cls,
        version: str,
        checks: dict[str, HealthStatus],
        **details: Any,
    ) -> "HealthCheck":
        """Create a degraded status response.

        Args:
            version: Application version string.
            checks: Component health check results.
            **details: Optional additional details.

        Returns:
            HealthCheck with degraded status.
        """
        return cls(
            status=HealthStatus.DEGRADED,
            version=version,
            checks=checks,
            details=details,
        )

    @classmethod
    def unhealthy(
        cls,
        version: str,
        checks: dict[str, HealthStatus],
        **details: Any,
    ) -> "HealthCheck":
        """Create an unhealthy status response.

        Args:
            version: Application version string.
            checks: Component health check results.
            **details: Optional additional details.

        Returns:
            HealthCheck with unhealthy status.
        """
        return cls(
            status=HealthStatus.UNHEALTHY,
            version=version,
            checks=checks,
            details=details,
        )
