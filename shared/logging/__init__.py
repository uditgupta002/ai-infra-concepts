"""Structured logging module for AI Infrastructure.

Provides production-ready logging with:
- JSON formatted output for log aggregation
- Correlation ID tracking for request tracing
- Performance timing decorators
- Contextual logging
"""

from shared.logging.logger import (
    get_logger,
    configure_logging,
    LogContext,
    with_correlation_id,
    log_execution_time,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "LogContext",
    "with_correlation_id",
    "log_execution_time",
]
