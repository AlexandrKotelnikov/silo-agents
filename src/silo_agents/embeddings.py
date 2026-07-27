from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol, cast

import httpx


class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Small deterministic embedding used for reproducible local experiments."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[\w%-]+", text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OllamaEmbedder:
    """Embedding client for Ollama's native /api/embed endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._dimensions: int | None = None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self.embed("dimension probe")
        assert self._dimensions is not None
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        response = self._client.post("/api/embed", json={"model": self.model, "input": text})
        response.raise_for_status()
        raw: Any = response.json()
        if not isinstance(raw, dict):
            raise ValueError("Ollama embed response must be an object")
        body = cast(dict[str, Any], raw)
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("Ollama embed response omitted embeddings")
        first = embeddings[0]
        if not isinstance(first, list) or not first:
            raise ValueError("Ollama returned an invalid embedding vector")
        vector = [float(value) for value in first]
        if self._dimensions is None:
            self._dimensions = len(vector)
        elif len(vector) != self._dimensions:
            raise ValueError("Ollama embedding dimensions changed during the run")
        return vector
