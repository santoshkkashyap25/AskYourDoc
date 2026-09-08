"""Persistent ChromaDB vector store implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.domain.models import DocumentChunk, RetrievedChunk
from src.core.interfaces.vector_store import BaseVectorStore

logger = logging.getLogger("AskMyPDF.ChromaStore")


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of BaseVectorStore for persistent on-disk embedding storage."""

    COLLECTION_NAME = "ask_my_pdf_documents"

    def __init__(self, persist_directory: Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Lazy-initialize ChromaDB persistent client and collection."""
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            logger.info("Initializing ChromaDB PersistentClient at '%s'...", self.persist_directory)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection '%s' ready. Current count: %d",
                        self.COLLECTION_NAME, self._collection.count())
        return self._collection

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add or update document chunks with embeddings in ChromaDB."""
        if not chunks:
            return

        collection = self._get_collection()

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk '{c.chunk_id}' has no embedding vector populated.")
            ids.append(c.chunk_id)
            embeddings.append(c.embedding)
            documents.append(c.text)
            metadatas.append(c.to_metadata_dict())

        # Batch upsert into ChromaDB
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

        logger.info("Upserted %d chunks into ChromaDB collection '%s'", len(chunks), self.COLLECTION_NAME)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Query ChromaDB for nearest chunks using cosine similarity."""
        collection = self._get_collection()
        total_items = collection.count()
        if total_items == 0:
            return []

        actual_k = min(top_k, total_items)
        where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[RetrievedChunk] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
            # Chroma with cosine distance returns values in [0, 2], where 0 is identical
            # Normalized similarity score = 1.0 - (dist / 2.0)
            similarity = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
            if similarity < min_score:
                continue

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", "unknown"),
                filename=meta.get("filename", "unknown"),
                page_number=meta.get("page_number", 1),
                chunk_index=meta.get("chunk_index", 0),
                text=doc_text,
                metadata=meta
            )
            retrieved.append(RetrievedChunk(chunk=chunk, similarity_score=similarity))

        # Sort descending by score
        retrieved.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieved

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        collection = self._get_collection()
        collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted all chunks for doc_id='%s' from ChromaDB", doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents with metadata from ChromaDB."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        all_meta = collection.get(include=["metadatas"])["metadatas"]
        docs_map: Dict[str, Dict[str, Any]] = {}

        for meta in all_meta:
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "total_pages": meta.get("total_pages", 1),
                    "chunk_count": 0
                }
            docs_map[doc_id]["chunk_count"] += 1

        return list(docs_map.values())
