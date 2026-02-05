"""API middleware for request processing.

Provides middleware for logging, correlation IDs, and error handling.
"""

import logging
import time
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to requests.

    Extracts correlation ID from headers or generates a new one,
    making it available throughout the request lifecycle.
    """

    HEADER_NAME = "X-Correlation-ID"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response with correlation ID header.
        """
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.HEADER_NAME)
        if not correlation_id:
            correlation_id = str(uuid4())

        # Store in request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers[self.HEADER_NAME] = correlation_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging.

    Logs request details and response timing.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process and log the request.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        start_time = time.perf_counter()

        # Get correlation ID if available
        correlation_id = getattr(request.state, "correlation_id", None)

        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "correlation_id": correlation_id,
                "client_host": request.client.host if request.client else None,
            },
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
            },
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling.

    Catches exceptions and returns structured error responses.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request with error handling.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response or error response.
        """
        try:
            return await call_next(request)
        except Exception as e:
            # Get correlation ID if available
            correlation_id = getattr(request.state, "correlation_id", None)

            logger.exception(
                "Unhandled exception",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "path": request.url.path,
                },
            )

            # Re-raise to let FastAPI handle it
            raise


def setup_middleware(app: ASGIApp) -> None:
    """Set up all middleware on the application.

    Args:
        app: FastAPI application instance.
    """
    from fastapi import FastAPI

    if isinstance(app, FastAPI):
        app.add_middleware(ErrorHandlingMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(CorrelationIdMiddleware)
