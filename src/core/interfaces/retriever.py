"""Abstract base class for document chunk retrievers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.domain.models import RetrievedChunk


class BaseRetriever(ABC):
    """Interface contract for retrieving relevant context chunks given a query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        filter_doc_id: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """Retrieve most relevant chunks for the given user query.

        Args:
            query: Natural language question.
            top_k: Number of chunks to retrieve.
            filter_doc_id: Optional filter for a specific document ID.

        Returns:
            List of RetrievedChunk instances.
        """
        pass
