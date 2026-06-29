"""BYO-LLM adapter tests (plan v2 Task 10) -- no network, HTTP monkeypatched."""

from __future__ import annotations

import srag.adapters as adapters
from srag.adapters import OpenAIChat, OllamaChat, VLLMChat


def test_openai_chat_builds_request_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout=60.0):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "hello world"}}]}

    monkeypatch.setattr(adapters, "_http_post_json", fake_post)
    chat = OpenAIChat(model="gpt-4o-mini", api_key="sk-test")
    out = chat("say hi")
    assert out == "hello world"
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["messages"][0]["content"] == "say hi"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_chat_custom_base_url(monkeypatch):
    seen = {}
    monkeypatch.setattr(adapters, "_http_post_json",
                        lambda url, p, h, timeout=60.0: seen.update(url=url) or
                        {"choices": [{"message": {"content": "x"}}]})
    OpenAIChat(model="m", base_url="https://host/v1/", api_key="k")("q")
    assert seen["url"] == "https://host/v1/chat/completions"


def test_vllm_chat_defaults_to_local_openai_api(monkeypatch):
    seen = {}
    monkeypatch.setattr(adapters, "_http_post_json",
                        lambda url, p, h, timeout=60.0: seen.update(url=url, headers=h) or
                        {"choices": [{"message": {"content": "ok"}}]})
    out = VLLMChat(model="Qwen/Qwen2.5-3B-Instruct")("hi")
    assert out == "ok"
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"


def test_ollama_chat_builds_request_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout=120.0):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "ollama says hi"}}

    monkeypatch.setattr(adapters, "_http_post_json", fake_post)
    out = OllamaChat(model="qwen2.5:3b-instruct")("hello")
    assert out == "ollama says hi"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["stream"] is False


def test_adapters_are_chat_callables_for_the_generator(monkeypatch):
    # An adapter must satisfy chat(prompt)->str so make_llm_generator_fn accepts it.
    from srag.llm import make_llm_generator_fn
    monkeypatch.setattr(adapters, "_http_post_json",
                        lambda url, p, h, timeout=60.0:
                        {"choices": [{"message": {"content":
                         '{"answer":"Paris","claims":[],"answerable":true}'}}]})
    gen_fn = make_llm_generator_fn(OpenAIChat(model="m", api_key="k"))
    out = gen_fn("q", [])
    assert isinstance(out, dict)
    assert out.get("answer") == "Paris"
