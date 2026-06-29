"""Use self-correcting RAG on YOUR OWN data in a few lines.

    python examples/quickstart.py

Swap `MY_DOCS` for your documents (a list of strings, or dicts with a "text"
field). By default this runs on deterministic offline components — no weights,
no network — so it works immediately. For production, pass your own LLM and
turn on real encoders (see the commented lines).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import SelfCorrectingRAG  # noqa: E402

# 1) YOUR DATA — just a list of strings (or {"text", "id", "source"} dicts).
MY_DOCS = [
    "Ran is a 1985 epic samurai film directed by Akira Kurosawa.",
    "Akira Kurosawa was a Japanese filmmaker and painter, born in Tokyo in 1910.",
    "Seven Samurai is a 1954 film directed by Akira Kurosawa.",
    "The Kintai Bridge is a wooden arch bridge in Iwakuni, Japan, built in 1673.",
    "Reciprocal Rank Fusion merges a sparse and a dense ranked list of documents.",
]

# 2) BUILD a self-correcting RAG over your data (offline by default).
rag = SelfCorrectingRAG.from_documents(MY_DOCS)
#   For production with your own model:
#   from srag import TransformersChat
#   rag = SelfCorrectingRAG.from_documents(MY_DOCS,
#                                          llm=TransformersChat("Qwen/Qwen2.5-3B-Instruct"),
#                                          real_models=True)
#   ...or bring any chat callable:  llm=lambda prompt: my_openai_call(prompt)

# 3) (optional) calibrate when to answer vs. abstain on a few labelled examples.
rag.calibrate([
    {"question": "Who directed Ran?", "answerable": True},
    {"question": "Where is the Kintai Bridge?", "answerable": True},
    {"question": "What is the capital of Mars?", "answerable": False},
    {"question": "Who won the 2050 World Cup?", "answerable": False},
])

# 4) ASK. You get a grounded answer, citations, the diagnosis trail, and an
#    honest abstention when the corpus can't support an answer.
for q in [
    "What nationality is the director of the film Ran?",   # multi-hop -> self-corrects
    "In what year was the Kintai Bridge built?",
    "Who won the 2050 World Cup?",                          # not in the data -> abstains
]:
    r = rag.ask(q)
    print(f"\nQ: {q}")
    print(f"  status:      {r.status}")
    print(f"  answer:      {r.message}")
    print(f"  citations:   {r.citations}")
    print(f"  corrections: {r.corrections}   diagnoses: {r.diagnoses}")


if __name__ == "__main__":
    pass
