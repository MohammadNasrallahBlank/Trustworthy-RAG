"""Tests for the Stage 6b active-retrieval gate (doc section 4.1)."""

from __future__ import annotations

import json
import os

from srag import RetrievalGate, Stage1Pipeline, Planner
from srag.gate import _asserts_fact

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")


def _load():
    with open(CORPUS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_should_retrieve_heuristics():
    g = RetrievalGate()
    assert g.should_retrieve("Who directed Ran?").retrieve is True          # entity
    assert g.should_retrieve("What were the latest results in 2025?").retrieve is True  # time
    assert g.should_retrieve("What is 2 + 2?").retrieve is False            # arithmetic
    assert g.should_retrieve("Hello there").retrieve is False               # greeting


def test_should_retrieve_reasons():
    g = RetrievalGate()
    assert g.should_retrieve("What is 2+2?").reason == "arithmetic/parametric"
    assert g.should_retrieve("Who directed Ran?").reason == "entity-heavy"


def test_classifier_override():
    g = RetrievalGate(classifier=lambda q: False)
    d = g.should_retrieve("Who directed Ran?")
    assert d.retrieve is False and d.reason == "classifier"


def test_flare_spans_selects_low_confidence_factual():
    spans = [
        ("Akira Kurosawa was born in 1910", 0.2),   # low conf + factual -> pick
        ("which is a thing", 0.1),                   # low conf but filler -> skip
        ("He directed Ran", 0.9),                    # high conf -> skip
    ]
    picked = RetrievalGate.flare_spans(spans, threshold=0.5)
    assert "Akira Kurosawa was born in 1910" in picked
    assert "He directed Ran" not in picked
    assert "which is a thing" not in picked


def test_asserts_fact_helper():
    assert _asserts_fact("born in 1910")            # number
    assert _asserts_fact("Akira Kurosawa")          # proper noun
    assert not _asserts_fact("is a the of")         # all filler


def test_pipeline_skips_retrieval_when_gate_says_no():
    pipe = Stage1Pipeline(gate=RetrievalGate(classifier=lambda q: False))
    pipe.index_documents(_load())
    state = pipe.run("What is 2+2?")
    events = [e["event"] for e in state.trace]
    assert "gate" in events
    assert "retrieve" not in events                 # retrieval was skipped
    gate_evt = next(e for e in state.trace if e["event"] == "gate")
    assert gate_evt["retrieve"] is False


def test_pipeline_retrieves_when_gate_says_yes():
    pipe = Stage1Pipeline(gate=RetrievalGate(classifier=lambda q: True))
    pipe.index_documents(_load())
    state = pipe.run("Who directed Seven Samurai?")
    events = [e["event"] for e in state.trace]
    assert "gate" in events and "retrieve" in events
    assert "Kurosawa" in (state.answer + state.explanation)
