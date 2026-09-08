"""Abstract base class for Large Language Models and generators."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional


class BaseLLM(ABC):
    """Interface contract for generative language models with sync and streaming support."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a complete text response synchronously.

        Args:
            prompt: User/context prompt.
            system_prompt: Optional system instructions.

        Returns:
            Generated text string.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Stream generated text tokens sequentially.

        Args:
            prompt: User/context prompt.
            system_prompt: Optional system instructions.

        Yields:
            Token chunks as strings.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the human-readable identifier for this LLM provider."""
        pass
