# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Embedding service for vector search queries."""

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Base exception for embedding service operations."""

    pass


class EmbeddingServiceConfigError(EmbeddingServiceError):
    """Configuration error for embedding service."""

    pass


class EmbeddingServiceAPIError(EmbeddingServiceError):
    """API call error for embedding service."""

    pass


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAI-compatible embedding API."""

    api_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        api_url = os.environ.get("EMBEDDING_API_URL")
        api_key = os.environ.get("EMBEDDING_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL")

        if not api_url:
            raise EmbeddingServiceConfigError(
                "EMBEDDING_API_URL environment variable is not set"
            )
        if not api_key:
            raise EmbeddingServiceConfigError(
                "EMBEDDING_API_KEY environment variable is not set"
            )
        if not model:
            raise EmbeddingServiceConfigError(
                "EMBEDDING_MODEL environment variable is not set"
            )

        return cls(api_url=api_url, api_key=api_key, model=model)


class EmbeddingService:
    """Service for generating embeddings using OpenAI-compatible API."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client = httpx.Client(timeout=30.0)

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a query text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        url = f"{self.config.api_url.rstrip('/')}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": text,
            "model": self.config.model,
        }

        try:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # OpenAI format response structure:
            # {
            #   "data": [{"embedding": [...], "index": 0}],
            #   "model": "...",
            #   "usage": {...}
            # }
            return data["data"][0]["embedding"]

        except httpx.HTTPStatusError as e:
            logger.error("Embedding API HTTP error: %s", e)
            raise EmbeddingServiceAPIError(
                f"Embedding API request failed with status {e.response.status_code}"
            )
        except httpx.RequestError as e:
            logger.error("Embedding API request error: %s", e)
            raise EmbeddingServiceAPIError(f"Failed to connect to embedding API: {e}")
        except (KeyError, IndexError) as e:
            logger.error("Unexpected embedding API response format: %s", e)
            raise EmbeddingServiceAPIError(
                "Unexpected response format from embedding API"
            )

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
