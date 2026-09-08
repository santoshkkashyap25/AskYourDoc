"""Abstract base class for text splitters/chunkers."""

from abc import ABC, abstractmethod
from typing import List
from src.core.domain.models import Document, DocumentChunk


class BaseTextSplitter(ABC):
    """Interface contract for splitting documents into chunks for embedding."""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split a list of Document pages into smaller, semantically coherent DocumentChunks.

        Args:
            documents: List of Document objects.

        Returns:
            List of DocumentChunk instances with assigned chunk indices and IDs.
        """
        pass
