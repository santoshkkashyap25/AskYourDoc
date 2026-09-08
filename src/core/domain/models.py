"""Domain models representing core RAG business entities."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class DocumentMetadata:
    """Metadata associated with an uploaded document."""
    doc_id: str
    filename: str
    file_size_bytes: int
    content_hash: str
    total_pages: int
    total_chunks: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Document:
    """Represents an ingested document page or section."""
    doc_id: str
    filename: str
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents an atomic text chunk extracted from a document for embedding."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to a flat dictionary for vector DBs."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            **{k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))}
        }


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector similarity search with relevance score."""
    chunk: DocumentChunk
    similarity_score: float

    @property
    def relevance_percentage(self) -> int:
        return int(round(self.similarity_score * 100))


@dataclass
class Citation:
    """Grounding citation detailing where an answer was derived."""
    doc_id: str
    filename: str
    page_number: int
    chunk_id: str
    excerpt: str
    similarity_score: float


@dataclass
class QueryResult:
    """The final result of a RAG query execution."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    is_cached: bool = False
    execution_time_ms: float = 0.0
    llm_provider: str = "unknown"
