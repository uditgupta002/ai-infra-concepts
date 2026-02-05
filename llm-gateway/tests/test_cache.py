"""Tests for caching layer."""

import pytest
import asyncio

from llm_gateway.cache.storage import (
    CacheEntry,
    CacheStats,
    InMemoryCacheStorage,
    cosine_similarity,
)
from llm_gateway.cache.semantic_cache import SemanticCache, CacheResult
from llm_gateway.cache.embeddings import MockEmbeddingService
from llm_gateway.providers.models import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    TokenUsage,
)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_different_length_vectors(self):
        """Different length vectors should return 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_not_expired_without_expiry(self):
        """Entry without expiry should not be expired."""
        import time

        entry = CacheEntry(
            key="test",
            value={"content": "test"},
            embedding=[0.1, 0.2],
            created_at=time.time(),
            expires_at=None,
        )
        assert not entry.is_expired

    def test_expired_after_expiry(self):
        """Entry should be expired after expires_at."""
        import time

        entry = CacheEntry(
            key="test",
            value={"content": "test"},
            embedding=[0.1, 0.2],
            created_at=time.time() - 100,
            expires_at=time.time() - 10,
        )
        assert entry.is_expired

    def test_to_dict_from_dict(self):
        """Should serialize and deserialize correctly."""
        import time

        entry = CacheEntry(
            key="test",
            value={"content": "test"},
            embedding=[0.1, 0.2],
            created_at=time.time(),
            hit_count=5,
        )

        data = entry.to_dict()
        restored = CacheEntry.from_dict(data)

        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.embedding == entry.embedding
        assert restored.hit_count == entry.hit_count


class TestInMemoryCacheStorage:
    """Tests for InMemoryCacheStorage."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Should store and retrieve entries."""
        storage = InMemoryCacheStorage()

        await storage.set(
            key="test",
            value={"content": "hello"},
            embedding=[0.1, 0.2, 0.3],
        )

        entry = await storage.get("test")
        assert entry is not None
        assert entry.value["content"] == "hello"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        """Should return None for missing entries."""
        storage = InMemoryCacheStorage()

        entry = await storage.get("nonexistent")
        assert entry is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Should delete entries."""
        storage = InMemoryCacheStorage()

        await storage.set(
            key="test",
            value={"content": "hello"},
            embedding=[0.1, 0.2, 0.3],
        )

        deleted = await storage.delete("test")
        assert deleted

        entry = await storage.get("test")
        assert entry is None

    @pytest.mark.asyncio
    async def test_find_similar(self):
        """Should find similar entries by embedding."""
        storage = InMemoryCacheStorage()

        # Add entry with known embedding
        await storage.set(
            key="test",
            value={"content": "hello"},
            embedding=[1.0, 0.0, 0.0],
        )

        # Search with similar embedding
        similar = await storage.find_similar(
            embedding=[0.99, 0.1, 0.0],
            threshold=0.9,
        )

        assert len(similar) == 1
        entry, similarity = similar[0]
        assert entry.key == "test"
        assert similarity > 0.9

    @pytest.mark.asyncio
    async def test_find_similar_no_match(self):
        """Should return empty list when no similar entries."""
        storage = InMemoryCacheStorage()

        await storage.set(
            key="test",
            value={"content": "hello"},
            embedding=[1.0, 0.0, 0.0],
        )

        # Search with very different embedding
        similar = await storage.find_similar(
            embedding=[0.0, 1.0, 0.0],
            threshold=0.9,
        )

        assert len(similar) == 0

    @pytest.mark.asyncio
    async def test_clear(self):
        """Should clear all entries."""
        storage = InMemoryCacheStorage()

        await storage.set("test1", {"content": "a"}, [0.1])
        await storage.set("test2", {"content": "b"}, [0.2])

        await storage.clear()

        entries = await storage.get_all_entries()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_stats(self):
        """Should track statistics."""
        storage = InMemoryCacheStorage()

        await storage.set("test", {"content": "hello"}, [0.1])
        await storage.get("test")  # Hit
        await storage.get("nonexistent")  # Miss

        stats = await storage.get_stats()
        assert stats.size == 1
        assert stats.hits == 1
        assert stats.misses == 1

    @pytest.mark.asyncio
    async def test_eviction(self):
        """Should evict entries when at capacity."""
        storage = InMemoryCacheStorage(max_size=2)

        await storage.set("test1", {"content": "a"}, [0.1])
        await storage.set("test2", {"content": "b"}, [0.2])
        await storage.set("test3", {"content": "c"}, [0.3])

        # Should have evicted one entry
        entries = await storage.get_all_entries()
        assert len(entries) == 2


class TestMockEmbeddingService:
    """Tests for MockEmbeddingService."""

    @pytest.mark.asyncio
    async def test_generates_embeddings(self):
        """Should generate embeddings."""
        service = MockEmbeddingService(dimension=768)

        embedding = await service.get_embedding("Hello, world!")
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self):
        """Same text should produce same embedding."""
        service = MockEmbeddingService()

        emb1 = await service.get_embedding("Hello, world!")
        emb2 = await service.get_embedding("Hello, world!")

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        service = MockEmbeddingService()

        emb1 = await service.get_embedding("Hello")
        emb2 = await service.get_embedding("Goodbye")

        assert emb1 != emb2


class TestSemanticCache:
    """Tests for SemanticCache."""

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Should return miss for uncached request."""
        embedding_service = MockEmbeddingService()
        cache = SemanticCache(
            embedding_service=embedding_service,
            similarity_threshold=0.95,
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        result = await cache.get(request)
        assert not result.hit
        assert result.response is None

    @pytest.mark.asyncio
    async def test_cache_hit_after_set(self):
        """Should return hit for cached request."""
        embedding_service = MockEmbeddingService()
        cache = SemanticCache(
            embedding_service=embedding_service,
            similarity_threshold=0.95,
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        response = CompletionResponse(
            id="resp-1",
            content="Hi there!",
            model="mock",
            provider="mock",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            request_id="test",
        )

        # Cache the response
        await cache.set(request, response)

        # Should get a hit
        result = await cache.get(request)
        assert result.hit
        assert result.response is not None
        assert result.response.content == "Hi there!"
        assert result.response.cached

    @pytest.mark.asyncio
    async def test_disabled_cache(self):
        """Should always miss when disabled."""
        cache = SemanticCache(enabled=False)

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        result = await cache.get(request)
        assert not result.hit

    @pytest.mark.asyncio
    async def test_skip_cache_flag(self):
        """Should skip cache when skip_cache is True."""
        embedding_service = MockEmbeddingService()
        cache = SemanticCache(
            embedding_service=embedding_service,
            similarity_threshold=0.95,
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
            skip_cache=True,
        )

        result = await cache.get(request)
        assert not result.hit

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Should invalidate cached entries."""
        embedding_service = MockEmbeddingService()
        cache = SemanticCache(
            embedding_service=embedding_service,
            similarity_threshold=0.95,
        )

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            request_id="test",
        )

        response = CompletionResponse(
            id="resp-1",
            content="Hi there!",
            model="mock",
            provider="mock",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            request_id="test",
        )

        # Cache and get the key
        cache_key = await cache.set(request, response)
        assert cache_key is not None

        # Invalidate
        await cache.invalidate(cache_key)

        # Should miss now
        result = await cache.get(request)
        assert not result.hit
