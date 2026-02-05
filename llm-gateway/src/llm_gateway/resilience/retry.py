"""Retry logic with exponential backoff.

Provides configurable retry policies for handling transient failures
in LLM requests.
"""

import asyncio
import functools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

from llm_gateway.providers.base import ProviderError, RateLimitError


T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.0, 0.5)
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (ProviderError,)
    )
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)

    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable.

        Args:
            exception: The exception to check.

        Returns:
            True if the exception should be retried.
        """
        # Check if it's a retryable exception type
        if not isinstance(exception, self.retryable_exceptions):
            return False

        # Check provider error retryable flag
        if isinstance(exception, ProviderError):
            if not exception.retryable:
                return False

            # Check status code if available
            if exception.status_code:
                return exception.status_code in self.retryable_status_codes

        return True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.

        Uses exponential backoff with optional jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter
        if self.jitter:
            jitter_factor = random.uniform(*self.jitter_range)
            delay = delay * (1 + jitter_factor)

        return delay

    def get_rate_limit_delay(self, exception: RateLimitError) -> float:
        """Get delay for rate limit errors.

        Uses retry-after header if available, otherwise uses backoff.

        Args:
            exception: The rate limit error.

        Returns:
            Delay in seconds.
        """
        if exception.retry_after:
            return exception.retry_after
        return self.max_delay


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any = None
    exception: Exception | None = None
    attempts: int = 0
    total_delay: float = 0.0


async def retry_with_policy(
    func: Callable[..., T],
    policy: RetryPolicy,
    *args: Any,
    **kwargs: Any,
) -> RetryResult:
    """Execute a function with retry policy.

    Args:
        func: Async function to execute.
        policy: Retry policy to use.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        RetryResult with outcome.
    """
    last_exception: Exception | None = None
    total_delay = 0.0

    for attempt in range(policy.max_attempts):
        try:
            result = await func(*args, **kwargs)
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                total_delay=total_delay,
            )
        except Exception as e:
            last_exception = e

            # Check if we should retry
            if not policy.is_retryable(e):
                return RetryResult(
                    success=False,
                    exception=e,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                )

            # Don't delay on last attempt
            if attempt < policy.max_attempts - 1:
                # Handle rate limit specially
                if isinstance(e, RateLimitError):
                    delay = policy.get_rate_limit_delay(e)
                else:
                    delay = policy.get_delay(attempt)

                total_delay += delay
                await asyncio.sleep(delay)

    return RetryResult(
        success=False,
        exception=last_exception,
        attempts=policy.max_attempts,
        total_delay=total_delay,
    )


async def with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    **kwargs: Any,
) -> T:
    """Execute a function with retries.

    Convenience function for simple retry scenarios.

    Args:
        func: Async function to execute.
        *args: Positional arguments for the function.
        max_attempts: Maximum retry attempts.
        base_delay: Base delay between retries.
        max_delay: Maximum delay between retries.
        **kwargs: Keyword arguments for the function.

    Returns:
        Result from the function.

    Raises:
        Exception: The last exception if all retries fail.
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    result = await retry_with_policy(func, policy, *args, **kwargs)

    if result.success:
        return result.result
    else:
        raise result.exception  # type: ignore


def retry_decorator(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[type[Exception]] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding retry logic to async functions.

    Args:
        max_attempts: Maximum retry attempts.
        base_delay: Base delay between retries.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter.
        retryable_exceptions: Exception types to retry.

    Returns:
        Decorator function.
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=tuple(retryable_exceptions) if retryable_exceptions else (ProviderError,),
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            result = await retry_with_policy(func, policy, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.exception  # type: ignore

        return wrapper  # type: ignore

    return decorator
