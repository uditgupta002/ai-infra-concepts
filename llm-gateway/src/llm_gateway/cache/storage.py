"""Cache storage backends.

Provides abstract interface and implementations for storing
cached LLM responses.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """A cached entry with metadata."""

    key: str
    value: dict[str, Any]
    embedding: list[float]
    created_at: float
    expires_at: float | None = None
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            embedding=data["embedding"],
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            hit_count=data.get("hit_count", 0),
        )


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    size: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheStorage(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        value: dict[str, Any],
        embedding: list[float],
        ttl: float | None = None,
    ) -> None:
        """Store a cache entry.

        Args:
            key: Cache key.
            value: Value to cache.
            embedding: Embedding vector for the key.
            ttl: Time-to-live in seconds.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted.
        """
        ...

    @abstractmethod
    async def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries.

        Returns:
            List of all cache entries.
        """
        ...

    @abstractmethod
    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.95,
        limit: int = 1,
    ) -> list[tuple[CacheEntry, float]]:
        """Find entries with similar embeddings.

        Args:
            embedding: Query embedding.
            threshold: Minimum similarity threshold.
            limit: Maximum entries to return.

        Returns:
            List of (entry, similarity) tuples.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        ...

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats instance.
        """
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity between -1 and 1.
    """
    import math

    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class InMemoryCacheStorage(CacheStorage):
    """In-memory cache storage.

    Suitable for development and single-instance deployments.
    """

    def __init__(
        self,
        max_size: int = 1000,
        cleanup_interval: float = 60.0,
    ) -> None:
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries.
            cleanup_interval: Seconds between cleanup runs.
        """
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop() -> None:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._evictions += 1

    async def _evict_lru(self) -> None:
        """Evict least recently used entries if at capacity."""
        while len(self._cache) >= self._max_size:
            # Find entry with lowest hit count
            if not self._cache:
                break

            lru_key = min(
                self._cache.keys(),
                key=lambda k: (self._cache[k].hit_count, self._cache[k].created_at),
            )
            del self._cache[lru_key]
            self._evictions += 1

    async def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hit_count += 1
            self._hits += 1
            return entry

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        embedding: list[float],
        ttl: float | None = None,
    ) -> None:
        """Store a cache entry."""
        async with self._lock:
            await self._evict_lru()

            now = time.time()
            expires_at = now + ttl if ttl else None

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                embedding=embedding,
                created_at=now,
                expires_at=expires_at,
            )

    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def get_all_entries(self) -> list[CacheEntry]:
        """Get all cache entries."""
        async with self._lock:
            return [e for e in self._cache.values() if not e.is_expired]

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.95,
        limit: int = 1,
    ) -> list[tuple[CacheEntry, float]]:
        """Find entries with similar embeddings."""
        async with self._lock:
            results: list[tuple[CacheEntry, float]] = []

            for entry in self._cache.values():
                if entry.is_expired:
                    continue

                similarity = cosine_similarity(embedding, entry.embedding)
                if similarity >= threshold:
                    results.append((entry, similarity))

            # Sort by similarity (descending) and limit
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            return CacheStats(
                size=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
