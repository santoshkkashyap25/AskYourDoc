"""Document loader implementation using pypdf and pdfplumber."""

import logging
from pathlib import Path
from typing import List
from src.core.domain.models import Document
from src.core.interfaces.loader import BaseDocumentLoader

logger = logging.getLogger("AskMyPDF.Loader")


class PyPDFLoader(BaseDocumentLoader):
    """Loads PDF documents using PyPDF with fallback to pdfplumber for resilient extraction."""

    def load(self, file_path: Path, doc_id: str) -> List[Document]:
        """Load and extract text page-by-page from a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents: List[Document] = []
        filename = file_path.name

        # Attempt extraction via pypdf first
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_num = page_idx + 1
                cleaned_text = page_text.strip()
                if cleaned_text:
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_num,
                            text=cleaned_text,
                            metadata={"total_pages": total_pages, "extractor": "pypdf"}
                        )
                    )
            
            if documents:
                logger.info("Loaded %d pages from '%s' using pypdf", len(documents), filename)
                return documents
        except Exception as e:
            logger.warning("pypdf extraction failed or was incomplete for '%s': %s. Trying pdfplumber fallback.", filename, e)

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                total_pages = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_num = page_idx + 1
                    cleaned_text = page_text.strip()
                    if cleaned_text:
                        documents.append(
                            Document(
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=cleaned_text,
                                metadata={"total_pages": total_pages, "extractor": "pdfplumber"}
                            )
                        )
            logger.info("Loaded %d pages from '%s' using pdfplumber", len(documents), filename)
            return documents
        except Exception as e:
            logger.error("All PDF extraction methods failed for '%s': %s", filename, e)
            raise RuntimeError(f"Failed to extract text from PDF '{filename}': {e}") from e
