"""Tests for resilience patterns."""

import pytest
import asyncio

from llm_gateway.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    CircuitBreakerRegistry,
)
from llm_gateway.resilience.retry import (
    RetryPolicy,
    retry_with_policy,
    with_retry,
    RetryResult,
)
from llm_gateway.resilience.rate_limiter import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    RateLimitExceeded,
)
from llm_gateway.providers.base import ProviderError


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit should open after failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        async def failing_func():
            raise Exception("Failure")

        # Cause failures
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_func)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_blocks_when_open(self):
        """Should block calls when circuit is open."""
        cb = CircuitBreaker(name="test", failure_threshold=1)

        async def failing_func():
            raise Exception("Failure")

        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)

        # Should now block
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(failing_func)

        assert exc_info.value.name == "test"

    @pytest.mark.asyncio
    async def test_allows_calls_when_closed(self):
        """Should allow calls when circuit is closed."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Reset should clear circuit state."""
        cb = CircuitBreaker(name="test", failure_threshold=1)

        async def failing_func():
            raise Exception("Failure")

        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)

        assert cb.is_open

        # Reset
        await cb.reset()
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_get_stats(self):
        """Should return circuit statistics."""
        cb = CircuitBreaker(name="test")
        stats = cb.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_creates_breaker(self):
        """Should create circuit breaker on first access."""
        registry = CircuitBreakerRegistry()
        cb = registry.get("test")

        assert cb is not None
        assert cb.name == "test"

    def test_get_returns_same_breaker(self):
        """Should return same breaker for same name."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get("test")
        cb2 = registry.get("test")

        assert cb1 is cb2

    def test_get_all(self):
        """Should return all breakers."""
        registry = CircuitBreakerRegistry()
        registry.get("test1")
        registry.get("test2")

        all_breakers = registry.get_all()
        assert len(all_breakers) == 2
        assert "test1" in all_breakers
        assert "test2" in all_breakers


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_is_retryable_provider_error(self):
        """Should identify retryable provider errors."""
        policy = RetryPolicy()

        error = ProviderError(
            message="Temporary failure",
            provider="test",
            status_code=500,
            retryable=True,
        )
        assert policy.is_retryable(error)

    def test_not_retryable_non_provider_error(self):
        """Should not retry non-provider errors."""
        policy = RetryPolicy()

        error = ValueError("Not a provider error")
        assert not policy.is_retryable(error)

    def test_not_retryable_when_flagged(self):
        """Should not retry when retryable=False."""
        policy = RetryPolicy()

        error = ProviderError(
            message="Auth failure",
            provider="test",
            status_code=401,
            retryable=False,
        )
        assert not policy.is_retryable(error)

    def test_get_delay_exponential(self):
        """Should calculate exponential backoff delay."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0

    def test_get_delay_capped(self):
        """Should cap delay at max_delay."""
        policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=False)

        assert policy.get_delay(10) == 5.0


class TestRetryWithPolicy:
    """Tests for retry_with_policy function."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Should return immediately on success."""
        policy = RetryPolicy(max_attempts=3)

        async def success_func():
            return "success"

        result = await retry_with_policy(success_func, policy)
        assert result.success
        assert result.result == "success"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Should retry on retryable failure."""
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError(
                    message="Failure",
                    provider="test",
                    status_code=500,
                    retryable=True,
                )
            return "success"

        result = await retry_with_policy(eventually_succeeds, policy)
        assert result.success
        assert result.result == "success"
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_returns_failure_after_max_attempts(self):
        """Should return failure after max attempts."""
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        async def always_fails():
            raise ProviderError(
                message="Always fails",
                provider="test",
                status_code=500,
                retryable=True,
            )

        result = await retry_with_policy(always_fails, policy)
        assert not result.success
        assert result.attempts == 3
        assert isinstance(result.exception, ProviderError)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Should allow requests within limit."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        result = await limiter.acquire("test")
        assert result.remaining >= 0
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Should block requests over limit."""
        limiter = TokenBucketRateLimiter(requests_per_minute=2, burst_size=2)

        # Use up the bucket
        await limiter.acquire("test")
        await limiter.acquire("test")

        # Should be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.acquire("test")

        assert exc_info.value.key == "test"
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_check_does_not_consume(self):
        """Check should not consume tokens."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        initial = await limiter.check("test")
        after_check = await limiter.check("test")

        assert initial.remaining == after_check.remaining

    @pytest.mark.asyncio
    async def test_reset_clears_limit(self):
        """Reset should clear rate limit for key."""
        limiter = TokenBucketRateLimiter(requests_per_minute=2, burst_size=2)

        # Use up the bucket
        await limiter.acquire("test")
        await limiter.acquire("test")

        # Reset
        await limiter.reset("test")

        # Should work again
        result = await limiter.acquire("test")
        assert result.remaining >= 0


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Should allow requests within limit."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=60)

        result = await limiter.acquire("test")
        assert result.remaining >= 0

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Should block requests over limit."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_size=60.0)

        # Use up the window
        await limiter.acquire("test")
        await limiter.acquire("test")

        # Should be blocked
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire("test")
