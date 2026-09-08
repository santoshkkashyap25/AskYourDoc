"""Document ingestion and lifecycle management service."""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.domain.models import Document, DocumentChunk, DocumentMetadata
from src.core.interfaces.embeddings import BaseEmbeddings
from src.core.interfaces.loader import BaseDocumentLoader
from src.core.interfaces.splitter import BaseTextSplitter
from src.core.interfaces.vector_store import BaseVectorStore
from src.core.interfaces.cache import BaseCache

logger = logging.getLogger("AskMyPDF.DocService")


class DocumentService:
    """Orchestrates document uploading, validation, parsing, chunking, and indexing."""

    def __init__(
        self,
        loader: BaseDocumentLoader,
        splitter: BaseTextSplitter,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        upload_dir: Path,
        max_documents: int = 10,
        max_total_storage_mb: int = 25,
        cache: Optional[BaseCache] = None
    ):
        self.loader = loader
        self.splitter = splitter
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_documents = max_documents
        self.max_total_storage_mb = max_total_storage_mb
        self.cache = cache
        self._documents_meta: Dict[str, DocumentMetadata] = {}

    def _compute_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Return current document count and storage usage metrics."""
        current_docs = self.list_documents()
        total_bytes = sum(f.stat().st_size for f in self.upload_dir.glob("*_*") if f.is_file())
        max_total_bytes = self.max_total_storage_mb * 1024 * 1024
        return {
            "total_documents": len(current_docs),
            "max_documents": self.max_documents,
            "total_bytes": total_bytes,
            "max_total_bytes": max_total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "max_total_mb": self.max_total_storage_mb
        }

    def ingest_pdf(self, filename: str, file_bytes: bytes) -> DocumentMetadata:
        """Ingest a PDF file from bytes, chunk it, embed it, and store in vector database."""
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files are supported. Received: {filename}")

        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        content_hash = self._compute_hash(file_bytes)

        # Check if identical file was already indexed
        for existing in self._documents_meta.values():
            if existing.content_hash == content_hash:
                logger.info("PDF '%s' with hash '%s' already indexed under doc_id='%s'",
                            filename, content_hash[:8], existing.doc_id)
                return existing

        # Constraint 1: Maximum document count limit
        current_docs = self.list_documents()
        if len(current_docs) >= self.max_documents:
            raise ValueError(
                f"Maximum document limit reached ({self.max_documents} documents). "
                f"Please delete existing documents to upload new ones."
            )

        # Constraint 2: Cumulative storage limit (max 25 MB across all documents)
        current_total_bytes = sum(f.stat().st_size for f in self.upload_dir.glob("*_*") if f.is_file())
        max_total_bytes = self.max_total_storage_mb * 1024 * 1024
        new_file_bytes = len(file_bytes)

        if current_total_bytes + new_file_bytes > max_total_bytes:
            current_mb = round(current_total_bytes / (1024 * 1024), 2)
            file_mb = round(new_file_bytes / (1024 * 1024), 2)
            remaining_mb = max(0.0, round((max_total_bytes - current_total_bytes) / (1024 * 1024), 2))
            raise ValueError(
                f"Total storage limit exceeded (maximum {self.max_total_storage_mb} MB). "
                f"Current usage: {current_mb} MB, new file: {file_mb} MB ({remaining_mb} MB remaining). "
                f"Please delete existing documents to free up space."
            )

        doc_id = str(uuid.uuid4())[:8]
        saved_path = self.upload_dir / f"{doc_id}_{filename}"
        saved_path.write_bytes(file_bytes)

        logger.info("Saved upload to '%s' (size: %d bytes)", saved_path, len(file_bytes))

        # 1. Load document pages
        pages: List[Document] = self.loader.load(saved_path, doc_id=doc_id)
        if not pages:
            raise ValueError(f"Could not extract any readable text from '{filename}'. It might be a scanned image.")

        # 2. Split into chunks
        chunks: List[DocumentChunk] = self.splitter.split_documents(pages)
        if not chunks:
            raise ValueError(f"No valid text chunks generated for '{filename}'.")

        # 3. Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings_list = self.embeddings.embed_documents(chunk_texts)
        for chunk, emb in zip(chunks, embeddings_list):
            chunk.embedding = emb

        # 4. Store in vector database
        self.vector_store.add_chunks(chunks)

        if self.cache:
            self.cache.clear("answers")
            logger.info("Invalidated 'answers' cache following ingestion of '%s'", filename)

        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            total_pages=len(pages),
            total_chunks=len(chunks)
        )
        self._documents_meta[doc_id] = meta

        logger.info("Successfully ingested '%s' (doc_id=%s, pages=%d, chunks=%d)",
                    filename, doc_id, len(pages), len(chunks))
        return meta

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents combining metadata and vector store counts."""
        vector_docs = self.vector_store.list_documents()
        docs_by_id = {d["doc_id"]: d for d in vector_docs}

        # Merge with in-memory metadata if available
        results: List[Dict[str, Any]] = []
        all_ids = set(docs_by_id.keys()) | set(self._documents_meta.keys())

        for d_id in all_ids:
            if d_id in self._documents_meta:
                m = self._documents_meta[d_id]
                results.append({
                    "doc_id": m.doc_id,
                    "filename": m.filename,
                    "file_size_bytes": m.file_size_bytes,
                    "total_pages": m.total_pages,
                    "total_chunks": m.total_chunks,
                    "created_at": m.created_at
                })
            elif d_id in docs_by_id:
                v = docs_by_id[d_id]
                results.append({
                    "doc_id": v["doc_id"],
                    "filename": v.get("filename", "Unknown"),
                    "file_size_bytes": 0,
                    "total_pages": v.get("total_pages", 1),
                    "total_chunks": v.get("chunk_count", 0),
                    "created_at": ""
                })

        return results

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector store, disk, and metadata cache."""
        self.vector_store.delete_document(doc_id)
        if doc_id in self._documents_meta:
            del self._documents_meta[doc_id]

        if self.cache:
            self.cache.clear("answers")
            logger.info("Invalidated 'answers' cache following deletion of doc_id='%s'", doc_id)

        # Clean up disk files
        for f in self.upload_dir.glob(f"{doc_id}_*"):
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", f, e)

        logger.info("Deleted document doc_id='%s'", doc_id)
        return True
