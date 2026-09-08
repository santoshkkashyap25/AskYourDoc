"""Dependency Injection Container (Service Factory) for AskMyPDF."""

import logging
from typing import Optional
from src.config import Settings, settings
from src.core.interfaces.cache import BaseCache
from src.core.interfaces.embeddings import BaseEmbeddings
from src.core.interfaces.loader import BaseDocumentLoader
from src.core.interfaces.llm import BaseLLM
from src.core.interfaces.splitter import BaseTextSplitter
from src.core.interfaces.vector_store import BaseVectorStore
from src.core.interfaces.workflow import BaseRAGWorkflow

from src.infrastructure.caching.sqlite_cache import SQLiteDiskCache
from src.infrastructure.loaders.pypdf_loader import PyPDFLoader
from src.infrastructure.splitters.recursive_splitter import RecursiveCharacterSplitter
from src.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from src.infrastructure.vector_stores.memory_store import MemoryVectorStore
from src.infrastructure.llm.mock_llm import MockExtractiveLLM
from src.infrastructure.llm.local_hf_llm import LocalHFLLM
from src.infrastructure.llm.openai_llm import OpenAILLM
from src.infrastructure.llm.gemini_llm import GeminiLLM
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.infrastructure.llm.groq_llm import GroqLLM
from src.infrastructure.llm.grok_llm import GrokLLM

logger = logging.getLogger("AskMyPDF.Container")


class Container:
    """Dependency Injection Container creating and wiring swappable components."""

    def __init__(self, app_settings: Optional[Settings] = None):
        self.settings = app_settings or settings
        self.settings.ensure_directories()

        # Component instances (lazily or eagerly initialized)
        self._cache: Optional[BaseCache] = None
        self._loader: Optional[BaseDocumentLoader] = None
        self._splitter: Optional[BaseTextSplitter] = None
        self._embeddings: Optional[BaseEmbeddings] = None
        self._vector_store: Optional[BaseVectorStore] = None
        self._llm: Optional[BaseLLM] = None
        self._workflow: Optional[BaseRAGWorkflow] = None
        self._document_service = None

    @property
    def cache(self) -> BaseCache:
        """Get or create the persistent cache layer."""
        if self._cache is None:
            logger.info("Instantiating SQLiteDiskCache at '%s'", self.settings.CACHE_DB_PATH)
            self._cache = SQLiteDiskCache(db_path=self.settings.CACHE_DB_PATH)
        return self._cache

    @property
    def loader(self) -> BaseDocumentLoader:
        """Get or create the document loader."""
        if self._loader is None:
            self._loader = PyPDFLoader()
        return self._loader

    @property
    def splitter(self) -> BaseTextSplitter:
        """Get or create the text splitter."""
        if self._splitter is None:
            self._splitter = RecursiveCharacterSplitter(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP
            )
        return self._splitter

    @property
    def embeddings(self) -> BaseEmbeddings:
        """Get or create the vector embedding provider."""
        if self._embeddings is None:
            logger.info("Initializing embedding provider: %s (model: %s)",
                        self.settings.EMBEDDING_PROVIDER, self.settings.EMBEDDING_MODEL_NAME)
            self._embeddings = SentenceTransformerEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL_NAME,
                cache=self.cache
            )
        return self._embeddings

    @property
    def vector_store(self) -> BaseVectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            try:
                import chromadb
                logger.info("Initializing persistent ChromaVectorStore at '%s'", self.settings.CHROMA_PERSIST_DIR)
                self._vector_store = ChromaVectorStore(persist_directory=self.settings.CHROMA_PERSIST_DIR)
            except (ImportError, Exception) as e:
                logger.warning("ChromaDB not available (%s). Falling back to in-memory vector store.", e)
                self._vector_store = MemoryVectorStore()
        return self._vector_store

    @property
    def llm(self) -> BaseLLM:
        """Get or create the LLM provider based on settings."""
        if self._llm is None:
            provider = self.settings.LLM_PROVIDER.lower()
            logger.info("Initializing LLM provider: '%s'", provider)

            if provider == "openai":
                if not self.settings.OPENAI_API_KEY:
                    logger.warning("OPENAI_API_KEY not set. Falling back to MockExtractiveLLM.")
                    self._llm = MockExtractiveLLM()
                else:
                    self._llm = OpenAILLM(
                        api_key=self.settings.OPENAI_API_KEY,
                        model_name=self.settings.OPENAI_MODEL_NAME
                    )
            elif provider == "gemini":
                if not self.settings.GEMINI_API_KEY:
                    logger.warning("GEMINI_API_KEY not set. Falling back to MockExtractiveLLM.")
                    self._llm = MockExtractiveLLM()
                else:
                    self._llm = GeminiLLM(
                        api_key=self.settings.GEMINI_API_KEY,
                        model_name=self.settings.GEMINI_MODEL_NAME
                    )
            elif provider == "ollama":
                self._llm = OllamaLLM(
                    base_url=self.settings.OLLAMA_BASE_URL,
                    model_name=self.settings.OLLAMA_MODEL_NAME
                )
            elif provider == "local":
                try:
                    self._llm = LocalHFLLM(model_name=self.settings.LOCAL_HF_MODEL_NAME)
                except Exception as e:
                    logger.warning("Local HF pipeline failed: %s. Falling back to MockExtractiveLLM.", e)
                    self._llm = MockExtractiveLLM()
            elif provider == "groq":
                if not self.settings.GROQ_API_KEY:
                    logger.warning("GROQ_API_KEY not set. Falling back to MockExtractiveLLM.")
                    self._llm = MockExtractiveLLM()
                else:
                    self._llm = GroqLLM(
                        api_key=self.settings.GROQ_API_KEY,
                        model_name=self.settings.GROQ_MODEL_NAME
                    )
            elif provider == "grok":
                api_key = self.settings.GROK_API_KEY
                if not api_key:
                    logger.warning("GROK_API_KEY not set. Falling back to MockExtractiveLLM.")
                    self._llm = MockExtractiveLLM()
                else:
                    self._llm = GrokLLM(
                        api_key=api_key,
                        model_name=self.settings.GROK_MODEL_NAME
                    )
            else:
                # Default / mock
                self._llm = MockExtractiveLLM()

            logger.info("LLM active provider: %s", self._llm.provider_name)
        return self._llm

    @property
    def workflow(self) -> BaseRAGWorkflow:
        """Get or create the LangGraph RAG workflow orchestrator."""
        if self._workflow is None:
            from src.services.workflow.langgraph_workflow import LangGraphRAGWorkflow
            self._workflow = LangGraphRAGWorkflow(
                vector_store=self.vector_store,
                embeddings=self.embeddings,
                llm=self.llm,
                cache=self.cache,
                top_k=self.settings.RETRIEVER_TOP_K,
                similarity_threshold=self.settings.SIMILARITY_THRESHOLD,
                cache_ttl=self.settings.CACHE_TTL_SECONDS
            )
        return self._workflow

    @property
    def document_service(self):
        """Get or create the document ingestion and management service."""
        if self._document_service is None:
            from src.services.document_service import DocumentService
            self._document_service = DocumentService(
                loader=self.loader,
                splitter=self.splitter,
                embeddings=self.embeddings,
                vector_store=self.vector_store,
                upload_dir=self.settings.UPLOAD_DIR,
                max_documents=self.settings.MAX_DOCUMENTS,
                max_total_storage_mb=self.settings.MAX_TOTAL_STORAGE_MB,
                cache=self.cache
            )
        return self._document_service


# Singleton container for application runtime
container = Container()
