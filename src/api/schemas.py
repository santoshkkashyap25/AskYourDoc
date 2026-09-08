"""Pydantic schemas for API request and response validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for incoming user questions."""
    question: str = Field(..., min_length=2, max_length=1000, description="The natural language question to ask")
    doc_id: Optional[str] = Field(None, description="Optional doc_id filter to scope search to a single document")


class CitationResponse(BaseModel):
    """Schema for individual document source citations."""
    doc_id: str
    filename: str
    page_number: int
    chunk_id: str
    excerpt: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Schema for full synchronous query answers."""
    question: str
    answer: str
    citations: List[CitationResponse] = []
    is_cached: bool = False
    execution_time_ms: float = 0.0
    llm_provider: str = "unknown"


class DocumentResponse(BaseModel):
    """Schema representing an indexed document."""
    doc_id: str
    filename: str
    file_size_bytes: int = 0
    total_pages: int = 1
    total_chunks: int = 0
    created_at: str = ""


class UploadResponse(BaseModel):
    """Schema returned after a successful PDF upload."""
    message: str
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int


class CacheStatsResponse(BaseModel):
    """Schema for cache monitoring statistics."""
    total_entries: int
    namespaces: Dict[str, int]
    file_size_kb: float
    hits: int
    misses: int
    hit_rate_percentage: float


class HealthResponse(BaseModel):
    """Schema for application health check."""
    status: str
    app_name: str
    llm_provider: str
    vector_store: str
    total_documents: int
