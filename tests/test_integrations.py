"""Integration-bridge tests (plan v2 Task 10d).

We don't install LangChain / LlamaIndex in CI, so we verify (a) the example
files compile, and (b) the wrapper PATTERN they document -- turning a framework
chat object into chat(prompt)->str -- actually drives SelfCorrectingRAG.
"""

from __future__ import annotations

import os
import py_compile

from srag import SelfCorrectingRAG

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, os.pardir, "examples", "integrations")


def test_integration_examples_compile():
    for name in ("langchain_example.py", "llamaindex_example.py"):
        py_compile.compile(os.path.join(EXP, name), doraise=True)


def test_langchain_style_wrapper_drives_srag():
    # Fake a LangChain chat model: .invoke(prompt).content
    class _Resp:
        def __init__(self, content):
            self.content = content

    class _FakeLC:
        def invoke(self, prompt):
            return _Resp('{"answer":"Paris","claims":'
                         '[{"text":"Paris is the capital.","citations":["d0"]}],'
                         '"answerable":true}')

    lc = _FakeLC()

    def chat(prompt: str) -> str:
        return lc.invoke(prompt).content

    rag = SelfCorrectingRAG.from_documents(
        [{"id": "d0", "text": "Paris is the capital of France."}], llm=chat)
    r = rag.ask("What is the capital of France?")
    assert "Paris" in (r.answer + r.message)


def test_llamaindex_style_doc_mapping():
    from examples.integrations import llamaindex_example as li  # type: ignore

    class _Doc:
        def __init__(self, id_, text):
            self.id_ = id_
            self._t = text

        def get_content(self):
            return self._t

    docs = li.docs_from_llamaindex([_Doc("kb-9", "hello world")])
    assert docs == [{"id": "kb-9", "text": "hello world"}]
