"""Local HuggingFace model implementation."""

import logging
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.LocalHFLLM")


class LocalHFLLM(BaseLLM):
    """Generative LLM using local Hugging Face transformers pipeline."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._pipeline = None

    @property
    def provider_name(self) -> str:
        return f"HuggingFace ({self.model_name})"

    def _get_pipeline(self):
        if self._pipeline is None:
            logger.info("Loading local HuggingFace pipeline '%s'...", self.model_name)
            from transformers import pipeline
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_new_tokens=256
            )
            logger.info("Local HuggingFace model loaded.")
        return self._pipeline

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pipe = self._get_pipeline()
        input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        results = pipe(input_text, max_new_tokens=256)
        return results[0]["generated_text"].strip()

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        full_text = self.generate(prompt, system_prompt, **kwargs)
        # Yield tokens sequentially
        for word in full_text.split(" "):
            yield word + " "
