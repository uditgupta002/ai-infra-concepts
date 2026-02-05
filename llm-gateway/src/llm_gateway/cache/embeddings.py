"""Embedding service for semantic caching.

Generates embeddings for cache key matching using
embedding models from configured providers.
"""

import hashlib
from typing import Any

import httpx

from llm_gateway.config import GatewaySettings, get_gateway_settings


class EmbeddingService:
    """Service for generating text embeddings.

    Uses OpenAI's embedding API by default for generating
    embeddings used in semantic cache matching.
    """

    def __init__(
        self,
        settings: GatewaySettings | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            settings: Gateway settings.
            model: Embedding model to use.
        """
        self._settings = settings or get_gateway_settings()
        self._model = model or self._settings.cache_embedding_model
        self._base_url = self._settings.openai_base_url

        # Cache of computed embeddings
        self._embedding_cache: dict[str, list[float]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        """Create HTTP client for embedding requests."""
        api_key = self._settings.openai_api_key
        if api_key is None:
            raise ValueError("OpenAI API key required for embeddings")

        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

    def _hash_text(self, text: str) -> str:
        """Create hash of text for caching."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text.

        Uses caching to avoid redundant API calls.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check cache first
        text_hash = self._hash_text(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Generate embedding
        embedding = await self._generate_embedding(text)

        # Cache result
        self._embedding_cache[text_hash] = embedding

        return embedding

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        async with self._get_client() as client:
            response = await client.post(
                "/embeddings",
                json={
                    "model": self._model,
                    "input": text,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Embedding request failed: {response.text}")

            data = response.json()
            return data["data"][0]["embedding"]

    async def get_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # Check which texts are already cached
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self._embedding_cache:
                results[i] = self._embedding_cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if uncached_texts:
            async with self._get_client() as client:
                response = await client.post(
                    "/embeddings",
                    json={
                        "model": self._model,
                        "input": uncached_texts,
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Batch embedding request failed: {response.text}")

                data = response.json()

                # Update results and cache
                for j, embedding_data in enumerate(data["data"]):
                    embedding = embedding_data["embedding"]
                    original_idx = uncached_indices[j]
                    results[original_idx] = embedding

                    # Cache the result
                    text_hash = self._hash_text(uncached_texts[j])
                    self._embedding_cache[text_hash] = embedding

        return [r for r in results if r is not None]  # type: ignore

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing.

    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536) -> None:
        """Initialize mock embedding service.

        Args:
            dimension: Embedding dimension.
        """
        self._dimension = dimension
        self._embedding_cache: dict[str, list[float]] = {}

    async def get_embedding(self, text: str) -> list[float]:
        """Generate mock embedding.

        Creates a deterministic embedding based on text hash.

        Args:
            text: Text to embed.

        Returns:
            Mock embedding vector.
        """
        text_hash = self._hash_text(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Generate deterministic embedding from hash
        import hashlib

        full_hash = hashlib.sha512(text.encode()).digest()

        # Convert hash bytes to floats
        embedding: list[float] = []
        for i in range(self._dimension):
            byte_idx = i % len(full_hash)
            value = (full_hash[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
            embedding.append(value)

        # Normalize the vector
        import math

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        self._embedding_cache[text_hash] = embedding
        return embedding

    def _hash_text(self, text: str) -> str:
        """Create hash of text for caching."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
