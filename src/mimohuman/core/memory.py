"""Pluggable long-term memory for agents."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single entry in a memory store."""

    key: str
    value: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: float | None = None


class MemoryStore(ABC):
    """Abstract interface for long-term memory storage.

    Implementations can back memory with in-process dicts, Redis,
    vector databases, etc.
    """

    @abstractmethod
    async def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value under the given key."""
        ...

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if not found or expired."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from the store."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Semantic search over stored entries. Default implementation returns empty."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries."""
        ...


class InMemoryStore(MemoryStore):
    """Simple dict-backed memory store. No persistence."""

    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}

    async def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._store[key] = MemoryEntry(key=key, value=value, ttl=ttl)

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.ttl is not None:
            age = (datetime.now(timezone.utc) - entry.timestamp).total_seconds()
            if age > entry.ttl:
                del self._store[key]
                return None
        return entry.value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        # Simple keyword match for now; swap in embeddings later
        results: list[MemoryEntry] = []
        query_lower = query.lower()
        for entry in self._store.values():
            if query_lower in entry.key.lower() or query_lower in str(entry.value).lower():
                results.append(entry)
            if len(results) >= limit:
                break
        return results

    async def clear(self) -> None:
        self._store.clear()
