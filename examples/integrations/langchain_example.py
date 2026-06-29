"""Use a LangChain chat model as the LLM for Self-Correcting RAG.

The bridge is trivial: any LangChain chat model exposes `.invoke(prompt)`, whose
result has a `.content` string. Wrap that into the `chat(prompt) -> str` contract
and pass it as `llm=`. Your retriever/vector store stays in LangChain if you
like; here we hand documents straight to srag, which does its own hybrid
retrieval + verification + abstention on top of the model.

Run:
    pip install langchain-openai
    OPENAI_API_KEY=... python examples/integrations/langchain_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srag import SelfCorrectingRAG


def langchain_chat(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Return a chat(prompt)->str callable backed by a LangChain chat model."""
    from langchain_openai import ChatOpenAI  # any langchain chat model works
    lc = ChatOpenAI(model=model, temperature=temperature)

    def chat(prompt: str) -> str:
        return lc.invoke(prompt).content

    return chat


DOCS = [
    {"id": "policy-1", "text": "Refunds are available within 30 days of purchase."},
    {"id": "policy-2", "text": "Shipping is free on orders over 50 dollars."},
]


def main():
    rag = SelfCorrectingRAG.from_documents(
        DOCS, llm=langchain_chat(), real_models=True, mode="trustworthy")
    for q in ["What is the refund window?", "Do you ship to Antarctica?"]:
        r = rag.ask(q)
        print(f"\nQ: {q}\n  {r.status}: {r.message}")


if __name__ == "__main__":
    main()
