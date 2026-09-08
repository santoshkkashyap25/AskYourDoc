"""Abstract base class for document loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from src.core.domain.models import Document


class BaseDocumentLoader(ABC):
    """Interface contract for loading documents from file or raw bytes."""

    @abstractmethod
    def load(self, file_path: Path, doc_id: str) -> List[Document]:
        """Load document from local file path and return structured pages.

        Args:
            file_path: Path to the PDF or document file.
            doc_id: Unique identifier for the document.

        Returns:
            List of Document objects representing pages/sections.
        """
        pass
