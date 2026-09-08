"""Abstract base class for vector embedding models."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Interface contract for calculating vector embeddings of text chunks and queries."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vector embeddings for a list of text strings.

        Args:
            texts: List of text chunk strings.

        Returns:
            List of float vectors of fixed dimension.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute dense vector embedding for a single user query.

        Args:
            text: Query string.

        Returns:
            Float vector of fixed dimension.
        """
        pass
