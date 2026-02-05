"""Rate limiting implementation using token bucket algorithm.

Provides both in-memory and Redis-backed rate limiters for
controlling request rates per key (user, org, API key).
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        key: str,
        limit: float,
        retry_after: float | None = None,
    ) -> None:
        """Initialize rate limit exceeded error.

        Args:
            key: The rate limit key that was exceeded.
            limit: The rate limit that was exceeded.
            retry_after: Optional seconds until retry is allowed.
        """
        self.key = key
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for key '{key}': {limit}/min")


@dataclass
class RateLimitInfo:
    """Information about current rate limit state."""

    remaining: float
    limit: float
    reset_at: float  # Unix timestamp
    retry_after: float | None = None

    @property
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        return self.remaining > 0


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(
        self,
        key: str,
        tokens: float = 1.0,
    ) -> RateLimitInfo:
        """Attempt to acquire tokens from the rate limiter.

        Args:
            key: Rate limit key (e.g., user ID, API key).
            tokens: Number of tokens to acquire.

        Returns:
            RateLimitInfo with current state.

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        ...

    @abstractmethod
    async def check(self, key: str) -> RateLimitInfo:
        """Check rate limit status without consuming tokens.

        Args:
            key: Rate limit key.

        Returns:
            RateLimitInfo with current state.
        """
        ...

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key to reset.
        """
        ...


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter with in-memory storage.

    The token bucket algorithm allows for burst traffic while
    maintaining an average rate limit.

    Attributes:
        rate: Tokens added per second.
        capacity: Maximum bucket capacity (burst size).
    """

    def __init__(
        self,
        requests_per_minute: float = 60.0,
        burst_size: float | None = None,
    ) -> None:
        """Initialize token bucket rate limiter.

        Args:
            requests_per_minute: Allowed requests per minute.
            burst_size: Maximum burst size. Defaults to requests_per_minute.
        """
        self._rate = requests_per_minute / 60.0  # Convert to per-second
        self._capacity = burst_size or requests_per_minute
        self._buckets: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, key: str) -> dict[str, float]:
        """Get or create a bucket for a key."""
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self._capacity,
                "last_update": now,
            }

        return self._buckets[key]

    def _refill_bucket(self, bucket: dict[str, float]) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket["last_update"]

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self._rate
        bucket["tokens"] = min(self._capacity, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

    async def acquire(
        self,
        key: str,
        tokens: float = 1.0,
    ) -> RateLimitInfo:
        """Attempt to acquire tokens.

        Args:
            key: Rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            RateLimitInfo with current state.

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        async with self._lock:
            bucket = self._get_bucket(key)
            self._refill_bucket(bucket)

            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return RateLimitInfo(
                    remaining=bucket["tokens"],
                    limit=self._capacity,
                    reset_at=time.time() + (self._capacity / self._rate),
                )
            else:
                # Calculate time until enough tokens available
                tokens_needed = tokens - bucket["tokens"]
                retry_after = tokens_needed / self._rate

                info = RateLimitInfo(
                    remaining=0,
                    limit=self._capacity,
                    reset_at=time.time() + retry_after,
                    retry_after=retry_after,
                )

                raise RateLimitExceeded(
                    key=key,
                    limit=self._capacity,
                    retry_after=retry_after,
                )

    async def check(self, key: str) -> RateLimitInfo:
        """Check rate limit status without consuming tokens.

        Args:
            key: Rate limit key.

        Returns:
            RateLimitInfo with current state.
        """
        async with self._lock:
            bucket = self._get_bucket(key)
            self._refill_bucket(bucket)

            return RateLimitInfo(
                remaining=bucket["tokens"],
                limit=self._capacity,
                reset_at=time.time() + (self._capacity / self._rate),
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key to reset.
        """
        async with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    async def wait_for_token(
        self,
        key: str,
        tokens: float = 1.0,
        timeout: float | None = None,
    ) -> RateLimitInfo:
        """Wait until tokens are available.

        Args:
            key: Rate limit key.
            tokens: Number of tokens needed.
            timeout: Maximum time to wait (None for no limit).

        Returns:
            RateLimitInfo after acquiring tokens.

        Raises:
            asyncio.TimeoutError: If timeout exceeded.
            RateLimitExceeded: If rate limit exceeded and can't wait.
        """
        start_time = time.time()

        while True:
            try:
                return await self.acquire(key, tokens)
            except RateLimitExceeded as e:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed

                    if remaining_timeout <= 0:
                        raise asyncio.TimeoutError("Timeout waiting for rate limit")

                    wait_time = min(e.retry_after or 1.0, remaining_timeout)
                else:
                    wait_time = e.retry_after or 1.0

                await asyncio.sleep(wait_time)


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter.

    More accurate than token bucket but uses more memory.
    Tracks individual request timestamps.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        window_size: float = 60.0,
    ) -> None:
        """Initialize sliding window rate limiter.

        Args:
            requests_per_minute: Maximum requests per window.
            window_size: Window size in seconds.
        """
        self._limit = requests_per_minute
        self._window_size = window_size
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    def _cleanup_old_requests(self, key: str) -> None:
        """Remove requests outside the current window."""
        if key not in self._requests:
            return

        cutoff = time.time() - self._window_size
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    async def acquire(
        self,
        key: str,
        tokens: float = 1.0,
    ) -> RateLimitInfo:
        """Attempt to acquire a request slot.

        Args:
            key: Rate limit key.
            tokens: Number of tokens (must be 1 for sliding window).

        Returns:
            RateLimitInfo with current state.

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        async with self._lock:
            if key not in self._requests:
                self._requests[key] = []

            self._cleanup_old_requests(key)
            now = time.time()

            current_count = len(self._requests[key])
            remaining = self._limit - current_count

            if remaining > 0:
                self._requests[key].append(now)
                return RateLimitInfo(
                    remaining=remaining - 1,
                    limit=self._limit,
                    reset_at=now + self._window_size,
                )
            else:
                # Calculate when oldest request will expire
                oldest = min(self._requests[key]) if self._requests[key] else now
                retry_after = oldest + self._window_size - now

                raise RateLimitExceeded(
                    key=key,
                    limit=self._limit,
                    retry_after=max(0, retry_after),
                )

    async def check(self, key: str) -> RateLimitInfo:
        """Check rate limit status.

        Args:
            key: Rate limit key.

        Returns:
            RateLimitInfo with current state.
        """
        async with self._lock:
            if key not in self._requests:
                self._requests[key] = []

            self._cleanup_old_requests(key)

            current_count = len(self._requests[key])
            remaining = max(0, self._limit - current_count)

            return RateLimitInfo(
                remaining=remaining,
                limit=self._limit,
                reset_at=time.time() + self._window_size,
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key to reset.
        """
        async with self._lock:
            if key in self._requests:
                del self._requests[key]


class CompositeRateLimiter(RateLimiter):
    """Combines multiple rate limiters.

    Useful for implementing multiple rate limit tiers
    (e.g., per-second, per-minute, per-day limits).
    """

    def __init__(self, limiters: list[RateLimiter]) -> None:
        """Initialize composite rate limiter.

        Args:
            limiters: List of rate limiters to combine.
        """
        self._limiters = limiters

    async def acquire(
        self,
        key: str,
        tokens: float = 1.0,
    ) -> RateLimitInfo:
        """Attempt to acquire tokens from all limiters.

        Args:
            key: Rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            RateLimitInfo with most restrictive state.

        Raises:
            RateLimitExceeded: If any rate limit is exceeded.
        """
        results = []
        for limiter in self._limiters:
            result = await limiter.acquire(key, tokens)
            results.append(result)

        # Return most restrictive result
        most_restrictive = min(results, key=lambda r: r.remaining)
        return most_restrictive

    async def check(self, key: str) -> RateLimitInfo:
        """Check all rate limiters.

        Args:
            key: Rate limit key.

        Returns:
            RateLimitInfo with most restrictive state.
        """
        results = []
        for limiter in self._limiters:
            result = await limiter.check(key)
            results.append(result)

        # Return most restrictive result
        return min(results, key=lambda r: r.remaining)

    async def reset(self, key: str) -> None:
        """Reset all rate limiters for a key.

        Args:
            key: Rate limit key to reset.
        """
        for limiter in self._limiters:
            await limiter.reset(key)
