"""Sentence-Transformers embedding model implementation."""

import hashlib
import logging
from typing import List, Optional
from src.core.interfaces.embeddings import BaseEmbeddings
from src.core.interfaces.cache import BaseCache

logger = logging.getLogger("AskMyPDF.Embeddings")


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Local, open-source embedding model using HuggingFace sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache: Optional[BaseCache] = None):
        self.model_name = model_name
        self.cache = cache
        self._model = None

    def _get_model(self):
        """Lazy load model on first use to speed up app boot time."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model '%s' loaded successfully.", self.model_name)
        return self._model

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute vector embeddings for multiple text chunks, leveraging cache where available."""
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # Check cache if cache is provided
        if self.cache:
            for idx, text in enumerate(texts):
                cache_key = self._hash_text(text)
                cached_vec = self.cache.get("embeddings", cache_key)
                if cached_vec is not None:
                    results[idx] = cached_vec
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(idx)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        # Compute embeddings for cache misses
        if texts_to_embed:
            model = self._get_model()
            logger.info("Computing embeddings for %d chunks...", len(texts_to_embed))
            computed_vectors = model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)

            for original_idx, text, vec in zip(indices_to_embed, texts_to_embed, computed_vectors):
                vec_list = vec.tolist()
                results[original_idx] = vec_list
                if self.cache:
                    cache_key = self._hash_text(text)
                    self.cache.set("embeddings", cache_key, vec_list)

        return results  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding vector for a single search query."""
        if self.cache:
            cache_key = self._hash_text(text)
            cached_vec = self.cache.get("embeddings", cache_key)
            if cached_vec is not None:
                return cached_vec

        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

        if self.cache:
            cache_key = self._hash_text(text)
            self.cache.set("embeddings", cache_key, vec)

        return vec
