"""Mock/Extractive local LLM for fast testing and zero-cost local execution."""

import logging
import re
import time
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.MockLLM")


class MockExtractiveLLM(BaseLLM):
    """Fast, deterministic, context-grounded extractive generator.

    Synthesizes concise, direct answers directly from provided context
    without needing external API keys or heavy GPU inference.
    """

    @property
    def provider_name(self) -> str:
        return "Local Grounded Engine (Extractive/Zero-Cost)"

    def _extract_answer_from_prompt(self, prompt: str) -> str:
        """Parse context from the prompt and craft a coherent, grounded response."""
        # Check if context was passed in the standard prompt template
        context_match = re.search(r"Context:\s*(.*?)\s*Question:", prompt, re.DOTALL | re.IGNORECASE)
        question_match = re.search(r"Question:\s*(.*?)$", prompt, re.DOTALL | re.IGNORECASE)

        context_text = context_match.group(1).strip() if context_match else prompt
        question = question_match.group(1).strip() if question_match else ""

        if not context_text or "No relevant context found" in context_text:
            return "Based on the uploaded documents, I could not find relevant information to answer your question."

        # Extract sentences from context
        raw_sentences = re.split(r"(?<=[.?!])\s+", context_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

        # Extract question keywords
        stop_words = {"what", "is", "the", "are", "how", "why", "when", "where", "who", "which", "a", "an", "in", "of", "to", "for", "on", "with", "does", "do", "can"}
        q_words = set(re.findall(r"\w+", question.lower())) - stop_words

        # Score sentences by keyword match
        scored = []
        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_words & s_words)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored[:3] if score > 0]

        if not top_sentences and sentences:
            top_sentences = sentences[:2]

        if not top_sentences:
            return "Based on the provided document excerpts, no definitive answer was found for this specific query."

        synthesized = " ".join(top_sentences)
        return f"Based on the uploaded document:\n\n{synthesized}"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        return self._extract_answer_from_prompt(prompt)

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        full_text = self._extract_answer_from_prompt(prompt)
        # Yield word-by-word with micro delays to simulate realistic token streaming
        words = full_text.split(" ")
        for i, word in enumerate(words):
            suffix = " " if i < len(words) - 1 else ""
            yield word + suffix
            time.sleep(0.015)
