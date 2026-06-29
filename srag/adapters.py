"""Bring-your-own-LLM chat adapters (plan v2 Task 10).

Every adapter is a callable `chat(prompt: str) -> str`, exactly the contract the
generator and planner expect (`make_llm_generator_fn` / `make_llm_planner_fn`).
Drop any of these into `SelfCorrectingRAG(..., llm=adapter)`.

  OpenAIChat   -- OpenAI / any OpenAI-compatible /chat/completions endpoint.
  OllamaChat   -- a local Ollama server (http://localhost:11434).
  VLLMChat     -- a local vLLM OpenAI-compatible server (http://localhost:8000/v1).
  TransformersChat -- re-exported from srag.llm (in-process Hugging Face model).

HTTP uses only the standard library (urllib), so there is no required runtime
dependency. All network I/O goes through the module-level `_http_post_json`,
which tests monkeypatch -- no network is touched in the suite.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

from .llm import TransformersChat  # re-export for a single import site

__all__ = ["OpenAIChat", "OllamaChat", "VLLMChat", "TransformersChat"]


def _http_post_json(url: str, payload: dict, headers: dict, timeout: float = 60.0) -> dict:
    """POST `payload` as JSON and parse the JSON response (stdlib only)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # pragma: no cover - network
        return json.loads(resp.read().decode("utf-8"))


class OpenAIChat:
    """OpenAI (or any OpenAI-compatible) chat-completions endpoint.

        chat = OpenAIChat(model="gpt-4o-mini")          # key from OPENAI_API_KEY
        chat = OpenAIChat(model="...", base_url="https://my-host/v1", api_key="x")
    """

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0, max_tokens: int = 512, timeout: float = 60.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = _http_post_json(f"{self.base_url}/chat/completions", payload, headers,
                               timeout=self.timeout)
        return resp["choices"][0]["message"]["content"]


class VLLMChat(OpenAIChat):
    """A local vLLM server (OpenAI-compatible API). Defaults to localhost:8000."""

    def __init__(self, model: str, *, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", **kwargs):
        super().__init__(model=model, base_url=base_url, api_key=api_key, **kwargs)


class OllamaChat:
    """A local Ollama server.

        chat = OllamaChat(model="qwen2.5:3b-instruct")
    """

    def __init__(self, model: str = "qwen2.5:3b-instruct", *,
                 host: str = "http://localhost:11434",
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = _http_post_json(f"{self.host}/api/chat", payload, {}, timeout=self.timeout)
        return resp["message"]["content"]
