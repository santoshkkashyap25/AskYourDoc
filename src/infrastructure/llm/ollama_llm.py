"""Ollama local server LLM adapter implementation."""

import json
import logging
from typing import Iterator, Optional
import httpx
from src.core.interfaces.llm import BaseLLM

logger = logging.getLogger("AskMyPDF.OllamaLLM")


class OllamaLLM(BaseLLM):
    """Generative LLM using a local Ollama server instance."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @property
    def provider_name(self) -> str:
        return f"Ollama ({self.model_name})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.2)
            }
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.2)
            }
        }
        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
