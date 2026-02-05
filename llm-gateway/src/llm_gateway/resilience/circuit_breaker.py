"""Circuit breaker pattern implementation.

Prevents cascading failures by temporarily stopping requests to
failing providers, allowing them time to recover.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


class CircuitOpenError(Exception):
    """Raised when circuit is open and blocking requests."""

    def __init__(
        self,
        name: str,
        time_until_retry: float,
        failure_count: int,
    ) -> None:
        """Initialize circuit open error.

        Args:
            name: Circuit breaker name.
            time_until_retry: Seconds until circuit will attempt recovery.
            failure_count: Number of failures that opened the circuit.
        """
        self.name = name
        self.time_until_retry = time_until_retry
        self.failure_count = failure_count
        super().__init__(
            f"Circuit '{name}' is open. Retry in {time_until_retry:.1f}s "
            f"(failures: {failure_count})"
        )


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    States:
        - CLOSED: Normal operation. Requests pass through. Failures are counted.
        - OPEN: Circuit is tripped. Requests are blocked immediately.
        - HALF_OPEN: Testing if service has recovered. Limited requests allowed.

    Transitions:
        - CLOSED -> OPEN: When failure threshold is exceeded.
        - OPEN -> HALF_OPEN: After recovery timeout expires.
        - HALF_OPEN -> CLOSED: When test requests succeed.
        - HALF_OPEN -> OPEN: When test requests fail.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3,
        failure_window: float = 60.0,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name (typically provider name).
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before attempting recovery.
            half_open_requests: Number of test requests in half-open state.
            failure_window: Time window for counting failures (seconds).
        """
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_requests = half_open_requests
        self._failure_window = failure_window

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._last_failure_time: datetime | None = None
        self._opened_at: datetime | None = None
        self._lock = asyncio.Lock()

        # Sliding window for failures
        self._failure_times: list[datetime] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _cleanup_old_failures(self) -> None:
        """Remove failures outside the failure window."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self._failure_window

        self._failure_times = [
            t for t in self._failure_times if t.timestamp() > cutoff
        ]
        self._failure_count = len(self._failure_times)

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._opened_at is None:
            return True

        now = datetime.now(timezone.utc)
        elapsed = (now - self._opened_at).total_seconds()
        return elapsed >= self._recovery_timeout

    def _time_until_retry(self) -> float:
        """Calculate time until circuit will attempt recovery."""
        if self._opened_at is None:
            return 0.0

        now = datetime.now(timezone.utc)
        elapsed = (now - self._opened_at).total_seconds()
        return max(0.0, self._recovery_timeout - elapsed)

    async def _record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1

                # If enough successes in half-open, close the circuit
                if self._half_open_successes >= self._half_open_requests:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._failure_times.clear()
                    self._half_open_successes = 0
                    self._opened_at = None

    async def _record_failure(self, error: Exception | None = None) -> None:
        """Record a failed request."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._last_failure_time = now
            self._failure_times.append(now)

            # Cleanup old failures
            self._cleanup_old_failures()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state, reopen the circuit
                self._state = CircuitState.OPEN
                self._opened_at = datetime.now(timezone.utc)
                self._half_open_successes = 0

            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = datetime.now(timezone.utc)

    async def _check_state(self) -> None:
        """Check and potentially transition circuit state."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0

    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result from the function.

        Raises:
            CircuitOpenError: If circuit is open.
            Exception: Any exception from the function.
        """
        await self._check_state()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(
                name=self.name,
                time_until_retry=self._time_until_retry(),
                failure_count=self._failure_count,
            )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    async def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_successes = 0
            self._failure_times.clear()
            self._opened_at = None
            self._last_failure_time = None

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit stats.
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": (
                self._last_failure_time.isoformat() if self._last_failure_time else None
            ),
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "time_until_retry": self._time_until_retry() if self.is_open else None,
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(
        self,
        default_failure_threshold: int = 5,
        default_recovery_timeout: float = 30.0,
        default_half_open_requests: int = 3,
    ) -> None:
        """Initialize registry.

        Args:
            default_failure_threshold: Default failure threshold.
            default_recovery_timeout: Default recovery timeout.
            default_half_open_requests: Default half-open requests.
        """
        self._default_failure_threshold = default_failure_threshold
        self._default_recovery_timeout = default_recovery_timeout
        self._default_half_open_requests = default_half_open_requests
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name.

        Returns:
            CircuitBreaker instance.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self._default_failure_threshold,
                recovery_timeout=self._default_recovery_timeout,
                half_open_requests=self._default_half_open_requests,
            )
        return self._breakers[name]

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers.

        Returns:
            Dictionary mapping names to circuit breakers.
        """
        return self._breakers.copy()

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers.

        Returns:
            Dictionary mapping names to stats.
        """
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
