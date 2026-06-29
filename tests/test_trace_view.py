"""Tests for the demo trace renderer."""
from __future__ import annotations

import json
import os

from srag import Controller, Planner, Stage1Pipeline, Verifier
from srag.trace_view import render_trace_html, render_trace_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")


def _state(q):
    docs = [json.loads(l) for l in open(CORPUS) if l.strip()]
    pipe = Stage1Pipeline(planner=Planner()).index_documents(docs)
    return Controller(pipe, Verifier(), tau=0.55, max_corrections=3).run(q)


def test_html_has_key_sections():
    h = render_trace_html(_state("What nationality is the director of the film Ran?"))
    for tok in ("QUESTION", "FINAL", "DECISION TRACE", "verify", "EVIDENCE"):
        assert tok in h


def test_markdown_renders_diagnosis_and_final():
    m = render_trace_markdown(_state("Who directed Seven Samurai?"))
    assert "diagnosis=" in m and "Final:" in m and "| step | detail |" in m
