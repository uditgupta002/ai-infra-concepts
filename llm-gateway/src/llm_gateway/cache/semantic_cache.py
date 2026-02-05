"""Semantic caching for LLM responses.

Caches responses based on semantic similarity of requests,
allowing cache hits for semantically equivalent but textually
different queries.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from llm_gateway.cache.storage import CacheStorage, InMemoryCacheStorage, CacheStats
from llm_gateway.cache.embeddings import EmbeddingService
from llm_gateway.providers.models import CompletionRequest, CompletionResponse


@dataclass
class CacheResult:
    """Result of a cache lookup."""

    hit: bool
    response: CompletionResponse | None = None
    similarity: float = 0.0
    cache_key: str | None = None


class SemanticCache:
    """Semantic cache for LLM responses.

    Uses embedding similarity to find cached responses for
    semantically similar requests, even if the exact wording differs.
    """

    def __init__(
        self,
        storage: CacheStorage | None = None,
        embedding_service: EmbeddingService | None = None,
        similarity_threshold: float = 0.95,
        ttl: float = 3600.0,
        enabled: bool = True,
    ) -> None:
        """Initialize semantic cache.

        Args:
            storage: Cache storage backend.
            embedding_service: Service for generating embeddings.
            similarity_threshold: Minimum similarity for cache hits.
            ttl: Cache TTL in seconds.
            enabled: Whether caching is enabled.
        """
        self._storage = storage or InMemoryCacheStorage()
        self._embedding_service = embedding_service
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled and self._embedding_service is not None

    def _get_cache_key_content(self, request: CompletionRequest) -> str:
        """Get content for cache key generation.

        Args:
            request: The completion request.

        Returns:
            Normalized content for cache key.
        """
        return request.get_cache_key_content()

    def _generate_cache_key(self, content: str) -> str:
        """Generate a cache key from content.

        Args:
            content: Content to hash.

        Returns:
            Cache key string.
        """
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get(self, request: CompletionRequest) -> CacheResult:
        """Look up a cached response.

        Args:
            request: The completion request.

        Returns:
            CacheResult with hit status and cached response if found.
        """
        if not self.enabled or request.skip_cache:
            return CacheResult(hit=False)

        if self._embedding_service is None:
            return CacheResult(hit=False)

        try:
            # Get content for cache lookup
            content = self._get_cache_key_content(request)

            # Generate embedding
            embedding = await self._embedding_service.get_embedding(content)

            # Search for similar cached entries
            similar = await self._storage.find_similar(
                embedding=embedding,
                threshold=self._similarity_threshold,
                limit=1,
            )

            if not similar:
                return CacheResult(hit=False, cache_key=self._generate_cache_key(content))

            entry, similarity = similar[0]

            # Reconstruct response from cached data
            response = CompletionResponse(**entry.value)
            response.cached = True

            return CacheResult(
                hit=True,
                response=response,
                similarity=similarity,
                cache_key=entry.key,
            )

        except Exception:
            # On any cache error, return miss and continue
            return CacheResult(hit=False)

    async def set(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
    ) -> str | None:
        """Cache a response.

        Args:
            request: The original request.
            response: The response to cache.

        Returns:
            Cache key if stored, None otherwise.
        """
        if not self.enabled or request.skip_cache:
            return None

        if self._embedding_service is None:
            return None

        try:
            # Get content and generate key
            content = self._get_cache_key_content(request)
            cache_key = self._generate_cache_key(content)

            # Generate embedding
            embedding = await self._embedding_service.get_embedding(content)

            # Prepare response data for caching
            response_data = response.model_dump()

            # Store in cache
            await self._storage.set(
                key=cache_key,
                value=response_data,
                embedding=embedding,
                ttl=self._ttl,
            )

            return cache_key

        except Exception:
            # On any cache error, return None and continue
            return None

    async def invalidate(self, cache_key: str) -> bool:
        """Invalidate a cached entry.

        Args:
            cache_key: Key of the entry to invalidate.

        Returns:
            True if entry was invalidated.
        """
        return await self._storage.delete(cache_key)

    async def clear(self) -> None:
        """Clear all cached entries."""
        await self._storage.clear()

    async def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats instance.
        """
        return await self._storage.get_stats()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable caching.

        Args:
            enabled: Whether to enable caching.
        """
        self._enabled = enabled

    def set_similarity_threshold(self, threshold: float) -> None:
        """Set similarity threshold for cache hits.

        Args:
            threshold: Minimum similarity (0-1).
        """
        self._similarity_threshold = max(0.0, min(1.0, threshold))

    def set_ttl(self, ttl: float) -> None:
        """Set cache TTL.

        Args:
            ttl: TTL in seconds.
        """
        self._ttl = max(0.0, ttl)
