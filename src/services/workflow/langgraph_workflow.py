"""LangGraph StateGraph workflow implementation for RAG orchestration."""

import hashlib
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, TypedDict
from src.core.domain.models import Citation, DocumentChunk, QueryResult, RetrievedChunk
from src.core.interfaces.cache import BaseCache
from src.core.interfaces.embeddings import BaseEmbeddings
from src.core.interfaces.llm import BaseLLM
from src.core.interfaces.vector_store import BaseVectorStore
from src.core.interfaces.workflow import BaseRAGWorkflow

logger = logging.getLogger("AskMyPDF.Workflow")


class RAGGraphState(TypedDict, total=False):
    """Internal state dictionary passed between nodes in the LangGraph graph."""
    question: str
    filter_doc_id: Optional[str]
    retrieved_chunks: List[RetrievedChunk]
    citations: List[Citation]
    context_prompt: str
    answer: str
    is_cached: bool
    execution_time_ms: float
    llm_provider: str


class LangGraphRAGWorkflow(BaseRAGWorkflow):
    """RAG workflow orchestrator combining LangGraph StateGraph with OOP interfaces."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        llm: BaseLLM,
        cache: BaseCache,
        top_k: int = 4,
        similarity_threshold: float = 0.25,
        cache_ttl: Optional[int] = 86400
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.cache = cache
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        self._graph = None

    def _cache_key(self, query: str, filter_doc_id: Optional[str]) -> str:
        raw = f"{query.strip().lower()}:{filter_doc_id or 'all'}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --- LangGraph Nodes ---

    def _check_cache_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Node 1: Inspect SQLite cache for pre-computed query answers."""
        key = self._cache_key(state["question"], state.get("filter_doc_id"))
        cached_data = self.cache.get("answers", key)
        if cached_data:
            logger.info("Cache HIT for query '%s'", state["question"][:40])
            citations = [
                Citation(
                    doc_id=c["doc_id"],
                    filename=c["filename"],
                    page_number=c["page_number"],
                    chunk_id=c["chunk_id"],
                    excerpt=c["excerpt"],
                    similarity_score=c["similarity_score"]
                )
                for c in cached_data.get("citations", [])
            ]
            return {
                "answer": cached_data["answer"],
                "citations": citations,
                "is_cached": True,
                "llm_provider": cached_data.get("llm_provider", self.llm.provider_name)
            }
        logger.info("Cache MISS for query '%s'", state["question"][:40])
        return {"is_cached": False}

    def _retrieve_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Node 2: Retrieve relevant chunks using embeddings and vector store."""
        question = state["question"]
        filter_doc_id = state.get("filter_doc_id")

        query_emb = self.embeddings.embed_query(question)
        retrieved = self.vector_store.similarity_search(
            query_embedding=query_emb,
            top_k=self.top_k,
            filter_doc_id=filter_doc_id,
            min_score=self.similarity_threshold
        )

        citations: List[Citation] = []
        for r in retrieved:
            chunk = r.chunk
            excerpt = (chunk.text[:200] + "...") if len(chunk.text) > 200 else chunk.text
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    excerpt=excerpt,
                    similarity_score=r.similarity_score
                )
            )

        return {
            "retrieved_chunks": retrieved,
            "citations": citations
        }

    def _grade_and_format_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Node 3: Grade context quality and construct grounded prompt."""
        retrieved = state.get("retrieved_chunks", [])
        question = state["question"]

        if not retrieved:
            context_str = "No relevant context found in the uploaded documents."
        else:
            context_parts = []
            for i, r in enumerate(retrieved, 1):
                c = r.chunk
                part = f"[Source {i} - Page {c.page_number} ({c.filename})]:\n{c.text}"
                context_parts.append(part)
            context_str = "\n\n".join(context_parts)

        prompt = (
            f"You are a helpful and accurate document assistant. Use ONLY the following context excerpts "
            f"from the uploaded PDF to answer the question. If the answer cannot be found in the context, "
            f"truthfully state that the document does not contain that information. Do not hallucinate.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        return {"context_prompt": prompt}

    def _generate_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Node 4: Synthesize response via BaseLLM."""
        prompt = state.get("context_prompt", state["question"])
        system_prompt = "You are a precise, grounded document question-answering assistant."
        answer = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return {
            "answer": answer,
            "llm_provider": self.llm.provider_name
        }

    def _cache_result_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Node 5: Persist freshly generated answer in the cache."""
        if not state.get("is_cached") and state.get("answer"):
            key = self._cache_key(state["question"], state.get("filter_doc_id"))
            serializable_citations = [
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_id": c.chunk_id,
                    "excerpt": c.excerpt,
                    "similarity_score": c.similarity_score
                }
                for c in state.get("citations", [])
            ]
            payload = {
                "answer": state["answer"],
                "citations": serializable_citations,
                "llm_provider": state.get("llm_provider", self.llm.provider_name)
            }
            self.cache.set("answers", key, payload, ttl=self.cache_ttl)
            logger.info("Saved query response to cache with key '%s'", key[:12])
        return {}

    def _compile_graph(self):
        """Compile LangGraph StateGraph if available, with resilient fallback."""
        if self._graph is not None:
            return self._graph

        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(RAGGraphState)
            workflow.add_node("check_cache", self._check_cache_node)
            workflow.add_node("retrieve", self._retrieve_node)
            workflow.add_node("grade_and_format", self._grade_and_format_node)
            workflow.add_node("generate", self._generate_node)
            workflow.add_node("cache_result", self._cache_result_node)

            workflow.set_entry_point("check_cache")

            def should_retrieve(state: RAGGraphState):
                return END if state.get("is_cached") else "retrieve"

            workflow.add_conditional_edges(
                "check_cache",
                should_retrieve,
                {"retrieve": "retrieve", END: END}
            )

            workflow.add_edge("retrieve", "grade_and_format")
            workflow.add_edge("grade_and_format", "generate")
            workflow.add_edge("generate", "cache_result")
            workflow.add_edge("cache_result", END)

            self._graph = workflow.compile()
            logger.info("LangGraph StateGraph compiled successfully.")
            return self._graph
        except Exception as e:
            logger.warning("Could not compile LangGraph (%s). Using native node dispatch.", e)
            return None

    def run(self, query: str, filter_doc_id: Optional[str] = None) -> QueryResult:
        """Execute full RAG workflow synchronously."""
        start_time = time.time()
        initial_state: RAGGraphState = {
            "question": query,
            "filter_doc_id": filter_doc_id,
            "is_cached": False,
            "citations": []
        }

        graph = self._compile_graph()
        if graph:
            final_state = graph.invoke(initial_state)
        else:
            # Native fallback sequential dispatch
            cache_res = self._check_cache_node(initial_state)
            initial_state.update(cache_res)
            if initial_state.get("is_cached"):
                final_state = initial_state
            else:
                ret_res = self._retrieve_node(initial_state)
                initial_state.update(ret_res)
                fmt_res = self._grade_and_format_node(initial_state)
                initial_state.update(fmt_res)
                gen_res = self._generate_node(initial_state)
                initial_state.update(gen_res)
                self._cache_result_node(initial_state)
                final_state = initial_state

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResult(
            question=query,
            answer=final_state.get("answer", "No response generated."),
            citations=final_state.get("citations", []),
            is_cached=final_state.get("is_cached", False),
            execution_time_ms=elapsed_ms,
            llm_provider=final_state.get("llm_provider", self.llm.provider_name)
        )

    def stream(self, query: str, filter_doc_id: Optional[str] = None) -> Iterator[Dict]:
        """Stream RAG response with immediate source citations and token-by-token emission."""
        start_time = time.time()
        key = self._cache_key(query, filter_doc_id)
        cached_data = self.cache.get("answers", key)

        if cached_data:
            # Emit cached result immediately
            citations = cached_data.get("citations", [])
            yield {"type": "sources", "citations": citations, "is_cached": True}
            yield {"type": "token", "content": cached_data["answer"]}
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            yield {
                "type": "done",
                "is_cached": True,
                "execution_time_ms": elapsed_ms,
                "llm_provider": cached_data.get("llm_provider", self.llm.provider_name)
            }
            return

        # Cache miss: retrieve first
        state: RAGGraphState = {"question": query, "filter_doc_id": filter_doc_id}
        ret_res = self._retrieve_node(state)
        state.update(ret_res)

        citations_list = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "excerpt": c.excerpt,
                "similarity_score": c.similarity_score
            }
            for c in state.get("citations", [])
        ]

        # Emit retrieved citations so the UI displays source badges right away
        yield {"type": "sources", "citations": citations_list, "is_cached": False}

        fmt_res = self._grade_and_format_node(state)
        prompt = fmt_res["context_prompt"]

        system_prompt = "You are a precise, grounded document question-answering assistant."
        generated_chunks: List[str] = []

        for token in self.llm.stream(prompt=prompt, system_prompt=system_prompt):
            generated_chunks.append(token)
            yield {"type": "token", "content": token}

        full_answer = "".join(generated_chunks).strip()
        state["answer"] = full_answer
        state["is_cached"] = False
        self._cache_result_node(state)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        yield {
            "type": "done",
            "is_cached": False,
            "execution_time_ms": elapsed_ms,
            "llm_provider": self.llm.provider_name
        }
