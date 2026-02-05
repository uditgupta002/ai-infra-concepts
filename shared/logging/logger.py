"""Structured JSON logging with correlation ID support.

This module provides production-ready logging capabilities including:
- JSON formatted logs for easy parsing and aggregation
- Correlation ID tracking across async operations
- Performance timing decorators
- Contextual logging with structured data
"""

import contextvars
import functools
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar
from uuid import uuid4

# Context variable for correlation ID tracking across async boundaries
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Produces JSON-formatted log records suitable for log aggregation
    systems like ELK, Splunk, or CloudWatch.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_correlation_id: bool = True,
    ) -> None:
        """Initialize the JSON formatter.

        Args:
            include_timestamp: Include ISO format timestamp.
            include_level: Include log level.
            include_logger: Include logger name.
            include_correlation_id: Include correlation ID if available.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_correlation_id = include_correlation_id

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Add correlation ID if available
        if self.include_correlation_id:
            correlation_id = _correlation_id.get()
            if correlation_id:
                log_data["correlation_id"] = correlation_id

        # Add message
        log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }
        }
        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development.

    Produces colored, readable log output suitable for local development.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as readable text.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        color = self.COLORS.get(record.levelname, "")
        correlation_id = _correlation_id.get()
        correlation_part = f"[{correlation_id[:8]}] " if correlation_id else ""

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"{color}{timestamp} | {record.levelname:8} | "
            f"{correlation_part}{record.name} | {record.getMessage()}{self.RESET}"
        )


class LogContext:
    """Context manager for adding structured context to logs.

    Usage:
        with LogContext(user_id="123", request_id="abc"):
            logger.info("Processing request")  # Includes user_id and request_id
    """

    _context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
        "log_context", default={}
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize log context with key-value pairs.

        Args:
            **kwargs: Key-value pairs to add to log context.
        """
        self.new_context = kwargs
        self.token: contextvars.Token[dict[str, Any]] | None = None

    def __enter__(self) -> "LogContext":
        """Enter the context, merging new context with existing."""
        current = self._context.get()
        merged = {**current, **self.new_context}
        self.token = self._context.set(merged)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context, restoring previous context."""
        if self.token:
            self._context.reset(self.token)

    @classmethod
    def get_current(cls) -> dict[str, Any]:
        """Get the current log context.

        Returns:
            Current context dictionary.
        """
        return cls._context.get()


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    logger_name: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Output format ('json' or 'text').
        logger_name: Specific logger to configure (None for root).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter based on format type
    if format_type.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    if logger_name:
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


@contextmanager
def with_correlation_id(correlation_id: str | None = None):
    """Context manager to set correlation ID for the current context.

    Args:
        correlation_id: Optional correlation ID. If None, generates a new UUID.

    Yields:
        The correlation ID being used.
    """
    cid = correlation_id or str(uuid4())
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


def get_correlation_id() -> str | None:
    """Get the current correlation ID.

    Returns:
        Current correlation ID or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> contextvars.Token[str | None]:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set.

    Returns:
        Token that can be used to reset the context.
    """
    return _correlation_id.set(correlation_id)


def log_execution_time(
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
    message: str = "Execution completed",
) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use. If None, creates one from function module.
        level: Log level for the timing message.
        message: Base message for the log.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message}: {func.__name__} took {elapsed:.2f}ms",
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message}: {func.__name__} failed after {elapsed:.2f}ms",
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message}: {func.__name__} took {elapsed:.2f}ms",
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{message}: {func.__name__} failed after {elapsed:.2f}ms",
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
