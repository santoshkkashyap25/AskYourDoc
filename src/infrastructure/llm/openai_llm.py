"""OpenAI API LLM adapter implementation."""

import logging
from typing import Iterator, Optional
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.OpenAILLM")


class OpenAILLM(BaseLLM):
    """Generative LLM using OpenAI's chat completion API."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided to use OpenAILLM")
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model_name})"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

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
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return response.choices[0].message.content or ""

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
            max_tokens=kwargs.get("max_tokens", 512),
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
