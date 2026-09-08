"""Recursive character text splitter implementation."""

import hashlib
import logging
from typing import List
from src.core.domain.models import Document, DocumentChunk
from src.core.interfaces.splitter import BaseTextSplitter

logger = logging.getLogger("AskMyPDF.Splitter")


class RecursiveCharacterSplitter(BaseTextSplitter):
    """Splits documents hierarchically using natural boundary separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str) -> List[str]:
        """Recursively split a text string into chunks not exceeding chunk_size."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Choose the first separator present in the text
        chosen_separator = ""
        for sep in self.separators:
            if sep == "":
                chosen_separator = ""
                break
            if sep in text:
                chosen_separator = sep
                break

        if chosen_separator != "":
            splits = text.split(chosen_separator)
        else:
            splits = list(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for s in splits:
            item_length = len(s) + (len(chosen_separator) if current_chunk else 0)
            if current_length + item_length <= self.chunk_size:
                current_chunk.append(s)
                current_length += item_length
            else:
                if current_chunk:
                    chunk_str = chosen_separator.join(current_chunk).strip()
                    if chunk_str:
                        chunks.append(chunk_str)

                # If a single split is still larger than chunk_size, recurse on it
                if len(s) > self.chunk_size:
                    sub_splits = self._split_text(s)
                    chunks.extend(sub_splits)
                    current_chunk = []
                    current_length = 0
                else:
                    # Calculate overlap from previous chunk if possible
                    current_chunk = [s]
                    current_length = len(s)

        if current_chunk:
            final_chunk = chosen_separator.join(current_chunk).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split a list of Document objects into indexed DocumentChunks."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for doc in documents:
            text_splits = self._split_text(doc.text)
            for split_idx, split_text in enumerate(text_splits):
                if not split_text.strip():
                    continue

                # Generate a deterministic chunk ID
                chunk_hash_input = f"{doc.doc_id}:{doc.page_number}:{split_idx}:{split_text[:32]}"
                chunk_id = hashlib.sha256(chunk_hash_input.encode("utf-8")).hexdigest()[:16]

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page_number=doc.page_number,
                    chunk_index=global_chunk_idx,
                    text=split_text,
                    metadata={
                        **doc.metadata,
                        "page_chunk_index": split_idx,
                        "char_count": len(split_text)
                    }
                )
                chunks.append(chunk)
                global_chunk_idx += 1

        logger.info("Split %d documents into %d chunks (chunk_size=%d, overlap=%d)",
                    len(documents), len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks
