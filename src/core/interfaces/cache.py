"""Abstract base class for caching systems."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseCache(ABC):
    """Interface contract for caching query answers, embeddings, and intermediate states."""

    @abstractmethod
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve a cached value by namespace and key.

        Args:
            namespace: Partition/namespace (e.g. 'embeddings', 'answers').
            key: Unique key within namespace.

        Returns:
            Cached value if found and not expired, else None.
        """
        pass

    @abstractmethod
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL in seconds.

        Args:
            namespace: Partition/namespace.
            key: Unique key within namespace.
            value: Serializable payload.
            ttl: Time-to-live in seconds.
        """
        pass

    @abstractmethod
    def delete(self, namespace: str, key: str) -> bool:
        """Delete a key from the cache."""
        pass

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all entries in a namespace or the entire cache if namespace is None."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return cache health, hit rate, entry counts, and memory/disk size."""
        pass
