"""In-memory NumPy-based vector store implementation."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from src.core.domain.models import DocumentChunk, RetrievedChunk
from src.core.interfaces.vector_store import BaseVectorStore

logger = logging.getLogger("AskMyPDF.MemoryStore")


class MemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store using NumPy for cosine similarity, ideal for unit testing and fallback."""

    def __init__(self):
        self._chunks: Dict[str, DocumentChunk] = {}
        self._vectors: Dict[str, np.ndarray] = {}

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk '{c.chunk_id}' has no embedding vector.")
            self._chunks[c.chunk_id] = c
            vec = np.array(c.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            self._vectors[c.chunk_id] = vec / (norm + 1e-9)
        logger.info("Added %d chunks to in-memory store. Total: %d", len(chunks), len(self._chunks))

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        if not self._chunks:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        q_vec = q_vec / (q_norm + 1e-9)

        results: List[RetrievedChunk] = []
        for chunk_id, vec in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if filter_doc_id and chunk.doc_id != filter_doc_id:
                continue

            score = float(np.dot(q_vec, vec))
            if score >= min_score:
                results.append(RetrievedChunk(chunk=chunk, similarity_score=score))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> None:
        to_delete = [c_id for c_id, c in self._chunks.items() if c.doc_id == doc_id]
        for c_id in to_delete:
            del self._chunks[c_id]
            del self._vectors[c_id]
        logger.info("Deleted %d chunks for doc_id='%s'", len(to_delete), doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for c in self._chunks.values():
            if c.doc_id not in docs:
                docs[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "total_pages": c.metadata.get("total_pages", 1),
                    "chunk_count": 0
                }
            docs[c.doc_id]["chunk_count"] += 1
        return list(docs.values())
