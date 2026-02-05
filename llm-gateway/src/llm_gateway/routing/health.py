"""Provider health tracking.

Tracks health metrics for providers including latency percentiles,
success rates, and error counts.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque
import statistics

from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.models import ProviderHealth
from llm_gateway.routing.strategies import ProviderMetrics


@dataclass
class LatencyMetrics:
    """Latency metrics with percentile calculations."""

    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self.samples.append(latency_ms)

    @property
    def p50(self) -> float:
        """50th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def mean(self) -> float:
        """Mean latency."""
        if not self.samples:
            return 0.0
        return statistics.mean(self.samples)


@dataclass
class SuccessMetrics:
    """Success rate tracking with sliding window."""

    window_size: int = 100
    successes: int = 0
    failures: int = 0
    _history: Deque[bool] = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self) -> None:
        """Record a successful request."""
        if len(self._history) >= self.window_size:
            oldest = self._history[0]
            if oldest:
                self.successes -= 1
            else:
                self.failures -= 1

        self._history.append(True)
        self.successes += 1

    def record_failure(self) -> None:
        """Record a failed request."""
        if len(self._history) >= self.window_size:
            oldest = self._history[0]
            if oldest:
                self.successes -= 1
            else:
                self.failures -= 1

        self._history.append(False)
        self.failures += 1

    @property
    def success_rate(self) -> float:
        """Current success rate."""
        total = self.successes + self.failures
        if total == 0:
            return 1.0
        return self.successes / total


class HealthTracker:
    """Tracks health metrics for multiple providers.

    Maintains latency, success rate, and error count metrics
    for each provider to inform routing decisions.
    """

    def __init__(
        self,
        health_check_interval: float = 30.0,
        unhealthy_threshold: float = 0.5,
    ) -> None:
        """Initialize health tracker.

        Args:
            health_check_interval: Seconds between health checks.
            unhealthy_threshold: Success rate below which provider is unhealthy.
        """
        self._health_check_interval = health_check_interval
        self._unhealthy_threshold = unhealthy_threshold

        self._latency_metrics: dict[str, LatencyMetrics] = {}
        self._success_metrics: dict[str, SuccessMetrics] = {}
        self._last_errors: dict[str, str] = {}
        self._last_health_checks: dict[str, datetime] = {}
        self._health_cache: dict[str, ProviderHealth] = {}
        self._request_counts: dict[str, int] = {}

        self._check_task: asyncio.Task | None = None
        self._providers: dict[str, LLMProvider] = {}

    def register_provider(self, provider: LLMProvider) -> None:
        """Register a provider for health tracking.

        Args:
            provider: Provider to track.
        """
        name = provider.name
        self._providers[name] = provider
        self._latency_metrics[name] = LatencyMetrics()
        self._success_metrics[name] = SuccessMetrics()
        self._request_counts[name] = 0

    def record_success(self, provider_name: str, latency_ms: float) -> None:
        """Record a successful request.

        Args:
            provider_name: Name of the provider.
            latency_ms: Request latency in milliseconds.
        """
        if provider_name in self._latency_metrics:
            self._latency_metrics[provider_name].record(latency_ms)

        if provider_name in self._success_metrics:
            self._success_metrics[provider_name].record_success()

        self._request_counts[provider_name] = (
            self._request_counts.get(provider_name, 0) + 1
        )

    def record_failure(self, provider_name: str, error: str) -> None:
        """Record a failed request.

        Args:
            provider_name: Name of the provider.
            error: Error message.
        """
        if provider_name in self._success_metrics:
            self._success_metrics[provider_name].record_failure()

        self._last_errors[provider_name] = error
        self._request_counts[provider_name] = (
            self._request_counts.get(provider_name, 0) + 1
        )

    def get_metrics(self, provider_name: str) -> ProviderMetrics:
        """Get metrics for a provider.

        Args:
            provider_name: Name of the provider.

        Returns:
            ProviderMetrics for the provider.
        """
        latency = self._latency_metrics.get(provider_name, LatencyMetrics())
        success = self._success_metrics.get(provider_name, SuccessMetrics())

        return ProviderMetrics(
            name=provider_name,
            is_healthy=self.is_healthy(provider_name),
            latency_p50_ms=latency.p50,
            latency_p95_ms=latency.p95,
            success_rate=success.success_rate,
            error_count=success.failures,
            request_count=self._request_counts.get(provider_name, 0),
        )

    def get_all_metrics(self) -> dict[str, ProviderMetrics]:
        """Get metrics for all tracked providers.

        Returns:
            Dictionary mapping provider names to metrics.
        """
        return {name: self.get_metrics(name) for name in self._providers}

    def is_healthy(self, provider_name: str) -> bool:
        """Check if a provider is healthy.

        Args:
            provider_name: Name of the provider.

        Returns:
            True if the provider is healthy.
        """
        success = self._success_metrics.get(provider_name)
        if success is None:
            return True  # Unknown providers are assumed healthy

        return success.success_rate >= self._unhealthy_threshold

    def get_health(self, provider_name: str) -> ProviderHealth:
        """Get health status for a provider.

        Args:
            provider_name: Name of the provider.

        Returns:
            ProviderHealth status.
        """
        latency = self._latency_metrics.get(provider_name, LatencyMetrics())
        success = self._success_metrics.get(provider_name, SuccessMetrics())

        return ProviderHealth(
            is_healthy=self.is_healthy(provider_name),
            latency_ms=latency.samples[-1] if latency.samples else None,
            latency_p50_ms=latency.p50,
            latency_p95_ms=latency.p95,
            latency_p99_ms=latency.p99,
            success_rate=success.success_rate,
            error_count=success.failures,
            last_error=self._last_errors.get(provider_name),
            last_check=self._last_health_checks.get(provider_name),
        )

    async def check_provider_health(self, provider: LLMProvider) -> ProviderHealth:
        """Perform a health check on a provider.

        Args:
            provider: Provider to check.

        Returns:
            ProviderHealth from the check.
        """
        health = await provider.health_check()
        self._last_health_checks[provider.name] = datetime.now(timezone.utc)
        self._health_cache[provider.name] = health

        # Update metrics based on health check result
        if health.is_healthy:
            if health.latency_ms:
                self.record_success(provider.name, health.latency_ms)
        else:
            self.record_failure(provider.name, health.last_error or "Health check failed")

        return health

    async def check_all_health(self) -> dict[str, ProviderHealth]:
        """Check health of all registered providers.

        Returns:
            Dictionary mapping provider names to health status.
        """
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await self.check_provider_health(provider)
            except Exception as e:
                results[name] = ProviderHealth(
                    is_healthy=False,
                    error_count=1,
                    last_error=str(e),
                    last_check=datetime.now(timezone.utc),
                )
        return results

    async def start_background_checks(self) -> None:
        """Start background health check task."""
        if self._check_task is not None:
            return

        async def check_loop() -> None:
            while True:
                try:
                    await self.check_all_health()
                except Exception:
                    pass  # Log but don't crash
                await asyncio.sleep(self._health_check_interval)

        self._check_task = asyncio.create_task(check_loop())

    async def stop_background_checks(self) -> None:
        """Stop background health check task."""
        if self._check_task is not None:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._latency_metrics.clear()
        self._success_metrics.clear()
        self._last_errors.clear()
        self._last_health_checks.clear()
        self._health_cache.clear()
        self._request_counts.clear()

        # Re-initialize for registered providers
        for name in self._providers:
            self._latency_metrics[name] = LatencyMetrics()
            self._success_metrics[name] = SuccessMetrics()
            self._request_counts[name] = 0
