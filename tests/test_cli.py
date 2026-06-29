"""CLI tests (plan v2 Task 10) -- run the parser/commands in-process."""

from __future__ import annotations

import json
import os

from srag.cli import main


_CORPUS = [
    {"id": "d0", "text": "The Eiffel Tower is located in Paris, the capital of France."},
    {"id": "d1", "text": "Mount Everest is the highest mountain on Earth."},
]


def _write_corpus(tmp_path):
    path = os.path.join(tmp_path, "corpus.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for d in _CORPUS:
            f.write(json.dumps(d) + "\n")
    return path


def test_index_then_ask(tmp_path, capsys):
    corpus = _write_corpus(tmp_path)
    idx = os.path.join(tmp_path, "index.pkl")
    rc = main(["index", corpus, "--out", idx])
    assert rc == 0 and os.path.exists(idx)

    rc = main(["ask", "Where is the Eiffel Tower located?", "--index", idx])
    assert rc == 0
    out = capsys.readouterr().out
    assert "status:" in out
    assert "Paris" in out


def test_ask_from_docs_json_output(tmp_path, capsys):
    corpus = _write_corpus(tmp_path)
    rc = main(["ask", "Where is the Eiffel Tower located?", "--docs", corpus, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"status", "answer", "citations"}


def test_ask_txt_corpus(tmp_path, capsys):
    path = os.path.join(tmp_path, "c.txt")
    open(path, "w", encoding="utf-8").write(
        "The capital of France is Paris.\nBerlin is the capital of Germany.\n")
    rc = main(["ask", "What is the capital of France?", "--docs", path])
    assert rc == 0
    assert "status:" in capsys.readouterr().out


def test_ask_requires_a_source(capsys):
    rc = main(["ask", "anything"])
    assert rc == 2


def test_evaluate_docs_offline(tmp_path, capsys):
    out = os.path.join(tmp_path, "eval")
    rc = main(["evaluate", "--dataset", "docs", "--backend", "offline", "--out", out])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "summary.json"))
