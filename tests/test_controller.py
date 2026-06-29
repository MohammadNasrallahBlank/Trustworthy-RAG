"""Tests for the Stage 4 controller: diagnosis -> correction + the control loop.

Offline (rule-based planner, lexical entailment, extractive generator). They
assert the routing behavior and the stopping conditions from doc sections
4.7-4.8, plus the headline result: the surgical bridge correction now produces
the right answer where Stage 1-3 only diagnosed the fault.
"""

from __future__ import annotations

import json
import os

import pytest

from srag import (
    Controller,
    EntailmentModel,
    GroundedGenerator,
    HybridRetriever,
    Embedder,
    CrossEncoderReranker,
    Planner,
    Stage1Pipeline,
    Verifier,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")


def load_corpus():
    with open(CORPUS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pipeline(planner=True):
    return Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=True)),
        reranker=CrossEncoderReranker(prefer_fallback=True),
        generator=GroundedGenerator(),
        planner=Planner() if planner else None,
    ).index_documents(load_corpus())


@pytest.fixture(scope="module")
def verifier():
    return Verifier(EntailmentModel(prefer_fallback=True))


# ----------------------------------- headline: bridge correction works
def test_bridge_correction_yields_right_answer(verifier):
    ctrl = Controller(build_pipeline(planner=True), verifier, tau=0.55, max_corrections=3)
    state = ctrl.run("What nationality is the director of the film Ran?")
    # The surgical fill-and-retrieve resolves the nationality.
    assert state.final_answer == "Japanese"
    assert state.answerable is True
    # At least one correction fired, and the loop recorded it.
    assert state.correction_count >= 1
    actions = [e for e in state.trace if e["event"] == "correct"]
    assert any(a["action"] == "fill-and-retrieve" for a in actions)
    # The kurosawa bio chunk is now cited.
    assert any("kurosawa-bio" in cid for cid in state.citations)


def test_bridge_correction_resolves_terminal_hop(verifier):
    ctrl = Controller(build_pipeline(planner=True), verifier)
    state = ctrl.run("What nationality is the director of the film Ran?")
    correct_events = [e for e in state.trace if e["event"] == "correct"]
    assert correct_events and correct_events[0]["action"] == "fill-and-retrieve"
    # The templated terminal hop was resolved with the bridge entity's name.
    terminal = [s for s in state.sub_queries if s.depends_on]
    assert terminal and terminal[0].resolved
    assert "Akira Kurosawa" in terminal[0].text


# ----------------------------------- pass-through (no correction)
def test_clean_question_passes_without_correction(verifier):
    ctrl = Controller(build_pipeline(planner=True), verifier)
    state = ctrl.run("Who directed Seven Samurai?")
    assert "Kurosawa" in state.final_answer
    assert state.correction_count == 0
    reasons = [e.get("reason") for e in state.trace if e["event"] == "control"]
    assert "pass" in reasons


# ----------------------------------- budget stop
def test_budget_caps_corrections(verifier):
    # An unanswerable question keeps diagnosing a fault; the budget must stop it.
    ctrl = Controller(build_pipeline(planner=True), verifier, tau=0.99, max_corrections=2)
    state = ctrl.run("What is the population of the Mars colony in 2200?")
    assert state.correction_count <= 2
    reasons = [e.get("reason") for e in state.trace if e["event"] == "control"]
    assert "budget_exhausted" in reasons or "no_new_evidence" in reasons


# ----------------------------------- no-new-evidence stop
def test_no_new_evidence_stops_loop(verifier):
    # High tau forces correction attempts; once retrieval can't surface anything
    # new, the loop must stop rather than spin to the budget cap every time.
    ctrl = Controller(build_pipeline(planner=False), verifier, tau=0.99, max_corrections=5)
    state = ctrl.run("What is the capital of France?")
    reasons = [e.get("reason") for e in state.trace if e["event"] == "control"]
    assert "no_new_evidence" in reasons
    # It stopped well before exhausting all 5 corrections.
    assert state.correction_count < 5


# ----------------------------------- conflict resolution by recency
def test_conflict_resolved_by_recency(verifier):
    docs = [
        {"id": "old", "source": "Old report", "timestamp": "2019-01-01",
         "text": "The Apex Bridge has a span of 300 meters across the river."},
        {"id": "new", "source": "New survey", "timestamp": "2025-01-01",
         "text": "The Apex Bridge does not have a span of 300 meters across the river; it spans 450 meters."},
    ]
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=True)),
        reranker=CrossEncoderReranker(prefer_fallback=True),
        generator=GroundedGenerator(),
    ).index_documents(docs)
    ctrl = Controller(pipe, verifier, tau=0.95, max_corrections=2)
    state = ctrl.run("What is the span of the Apex Bridge?")
    actions = [e["action"] for e in state.trace if e["event"] == "correct"]
    assert "resolve-conflict" in actions
    # Resolution note references the more recent source.
    assert "recency" in state.explanation.lower()


# ----------------------------------- web fallback hook
def test_web_fallback_invoked_when_corpus_exhausted(verifier):
    calls = {"n": 0}

    def fake_web(query):
        calls["n"] += 1
        return [{
            "id": "web-fr", "source": "web",
            "text": "Paris is the capital of France.", "rerank_score": 0.9,
        }]

    ctrl = Controller(
        build_pipeline(planner=False), verifier,
        tau=0.99, max_corrections=4, web_search=fake_web,
    )
    state = ctrl.run("What is the capital of France?")
    # The escalation ladder eventually reached the web fallback.
    assert calls["n"] >= 1
    assert any("web-fr" in cid for cid in state.seen_chunk_ids)
