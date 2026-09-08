"""Google Gemini API LLM adapter implementation."""

import logging
from typing import Iterator, Optional
import httpx
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.GeminiLLM")


class GeminiLLM(BaseLLM):
    """Generative LLM using Google Gemini REST API (gemini-1.5-flash / pro)."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided to use GeminiLLM")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    @property
    def provider_name(self) -> str:
        return f"Google Gemini ({self.model_name})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}?key={self.api_key}"
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
                "maxOutputTokens": kwargs.get("max_tokens", 512),
            }
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            logger.error("Failed to parse Gemini response: %s", data)
            raise RuntimeError(f"Unexpected response structure from Gemini API: {e}") from e

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        # Fallback to single generation chunked into tokens
        full_text = self.generate(prompt, system_prompt, **kwargs)
        for word in full_text.split(" "):
            yield word + " "
