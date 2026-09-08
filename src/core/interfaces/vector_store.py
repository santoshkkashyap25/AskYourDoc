"""Abstract base class for vector databases and similarity indexing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.core.domain.models import DocumentChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Interface contract for storing and querying document chunks with vector embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Insert or update document chunks with their embeddings in the vector index.

        Args:
            chunks: List of DocumentChunk instances with populated embeddings.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Perform cosine or distance-based nearest neighbor search.

        Args:
            query_embedding: Dense embedding vector of user query.
            top_k: Number of nearest chunks to retrieve.
            filter_doc_id: If specified, restrict search to this specific document ID.
            min_score: Minimum similarity score threshold.

        Returns:
            List of RetrievedChunk instances sorted by relevance score descending.
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks associated with a specific document ID.

        Args:
            doc_id: Unique document identifier.
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents indexed in the vector store with summary counts.

        Returns:
            List of dicts containing doc_id, filename, chunk_count, etc.
        """
        pass
