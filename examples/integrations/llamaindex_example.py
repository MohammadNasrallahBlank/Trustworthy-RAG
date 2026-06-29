"""Use a LlamaIndex LLM as the model for Self-Correcting RAG, and feed it
LlamaIndex documents/nodes.

Two bridges:
  * LLM:   any LlamaIndex LLM has `.complete(prompt)`, whose result has `.text`.
           Wrap it into chat(prompt)->str.
  * Data:  LlamaIndex `Document`/`Node` objects expose `.get_content()` and an
           id; map them onto srag's {"id","text"} documents.

Run:
    pip install llama-index-llms-openai llama-index-core
    OPENAI_API_KEY=... python examples/integrations/llamaindex_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srag import SelfCorrectingRAG


def llamaindex_chat(model: str = "gpt-4o-mini"):
    """Return a chat(prompt)->str callable backed by a LlamaIndex LLM."""
    from llama_index.llms.openai import OpenAI
    llm = OpenAI(model=model)

    def chat(prompt: str) -> str:
        return llm.complete(prompt).text

    return chat


def docs_from_llamaindex(nodes) -> list:
    """Map LlamaIndex Documents/Nodes onto srag documents."""
    out = []
    for i, n in enumerate(nodes):
        text = n.get_content() if hasattr(n, "get_content") else str(n)
        out.append({"id": getattr(n, "node_id", getattr(n, "id_", f"node{i}")),
                    "text": text})
    return out


def main():
    # Stand-in for nodes you'd load via a LlamaIndex reader/index.
    class _Doc:
        def __init__(self, id_, text):
            self.id_ = id_
            self._t = text

        def get_content(self):
            return self._t

    nodes = [
        _Doc("kb-1", "The warranty covers manufacturing defects for two years."),
        _Doc("kb-2", "Batteries are excluded from the standard warranty."),
    ]
    rag = SelfCorrectingRAG.from_documents(
        docs_from_llamaindex(nodes), llm=llamaindex_chat(), real_models=True)
    for q in ["How long is the warranty?", "Is the screen waterproof?"]:
        r = rag.ask(q)
        print(f"\nQ: {q}\n  {r.status}: {r.message}")


if __name__ == "__main__":
    main()
