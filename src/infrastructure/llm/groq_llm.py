"""Groq Cloud API LLM adapter implementation (Fast & Free Tier)."""

import logging
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.GroqLLM")


class GroqLLM(BaseLLM):
    """Generative LLM using Groq Cloud's ultra-fast free LPU inference API.
    
    Supports models like llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.
    """

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided to use GroqLLM")
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def provider_name(self) -> str:
        return f"Groq ({self.model_name})"

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _clean_response(self, text: str) -> str:
        """Strip internal thinking/reasoning tags if present in model output."""
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 500),
        )
        raw_text = response.choices[0].message.content or ""
        return self._clean_response(raw_text)

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 500),
            stream=True
        )
        inside_think = False
        buffer = ""

        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue

            buffer += delta
            if "<think>" in buffer:
                inside_think = True
            if "</think>" in buffer:
                inside_think = False
                buffer = buffer.split("</think>")[-1]
                continue

            if not inside_think:
                yield delta
