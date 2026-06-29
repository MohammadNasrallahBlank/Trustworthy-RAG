"""Tests for the Stage 3 planner and per-hop coverage integration.

Offline (rule-based planner, lexical entailment). They assert the decomposition
*contract* from doc section 4.2, and the headline Stage-3 result: once a bridge
question is decomposed, the verifier's coverage check (4) flags the unresolved
second hop instead of waving the whole question through as a single hop.
"""

from __future__ import annotations

import json
import os

import pytest

from srag import (
    EntailmentModel,
    Planner,
    Stage1Pipeline,
    Verifier,
    detect_question_type,
)
from srag.planner import _split_comparison

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")


def load_corpus():
    with open(CORPUS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------- type detection
@pytest.mark.parametrize(
    "q,expected",
    [
        ("What nationality is the director of the film Ran?", "bridge"),
        ("Where was the director of Ran born?", "bridge"),
        ("Who is older, Akira Kurosawa or Steven Spielberg?", "comparison"),
        ("Which film is longer, Ran or Seven Samurai?", "comparison"),
        ("Is the Kintai Bridge made of wood?", "yes-no"),
        ("Who directed Seven Samurai?", "single-hop"),
    ],
)
def test_detect_question_type(q, expected):
    assert detect_question_type(q) == expected


# -------------------------------------------------- bridge decomposition
def test_bridge_decomposition_two_hops_with_template():
    p = Planner()
    qtype, subs = p.plan("What nationality is the director of the film Ran?")
    assert qtype == "bridge"
    assert len(subs) == 2
    # Hop 0 resolves the bridge entity; it names the entity explicitly.
    assert "director" in subs[0].text.lower() and "ran" in subs[0].text.lower()
    # Hop 1 depends on hop 0 and is templated/unresolved until filled.
    assert subs[1].depends_on == "hop_0"
    assert subs[1].resolved is False
    assert "{entity}" in subs[1].template
    # Filling the template yields a name-explicit standalone query.
    filled = subs[1].fill("Akira Kurosawa")
    assert filled.resolved is True
    assert "Akira Kurosawa" in filled.text


# ---------------------------------------------- comparison decomposition
def test_comparison_splits_into_entities():
    p = Planner()
    qtype, subs = p.plan("Who is older, Akira Kurosawa or Steven Spielberg?")
    assert qtype == "comparison"
    assert len(subs) == 2
    joined = " ".join(s.text for s in subs)
    assert "Kurosawa" in joined and "Spielberg" in joined
    # Complementary, not paraphrases: the two hops differ.
    assert subs[0].text != subs[1].text


def test_split_comparison_helper():
    ents, pred = _split_comparison("Which film is longer, Ran or Seven Samurai?")
    assert ents == ["Ran", "Seven Samurai"]


# -------------------------------------------- planner never answers
def test_planner_never_answers():
    p = Planner()
    # The planner restructures the question; it must not inject the *answer*.
    # 'Japanese' is the answer to the Ran bridge and must never appear.
    _, ran_subs = p.plan("What nationality is the director of the film Ran?")
    assert all("japanese" not in s.text.lower() for s in ran_subs)

    # Every sub-query stays grounded in the original question's entities
    # (name-explicit) rather than becoming a free-standing answer.
    cases = {
        "Who is older, Akira Kurosawa or Steven Spielberg?": ["kurosawa", "spielberg"],
        "What nationality is the director of the film Ran?": ["ran"],
        "Who directed Seven Samurai?": ["samurai"],
    }
    for q, must_appear in cases.items():
        _, subs = p.plan(q)
        joined = " ".join(s.text for s in subs).lower()
        for token in must_appear:
            assert token in joined


def test_single_hop_passthrough():
    p = Planner()
    qtype, subs = p.plan("Who directed Seven Samurai?")
    assert qtype == "single-hop"
    assert len(subs) == 1
    assert subs[0].text == "Who directed Seven Samurai?"


# ---------------------------------------------- LLM planner path
def test_llm_planner_path_coerced_and_deduped():
    def fake_llm(q):
        return {
            "type": "comparison",
            "sub_queries": [
                {"text": "How old is Kurosawa?"},
                "How old is Kurosawa?",   # duplicate -> deduped
                {"text": "How old is Spielberg?"},
            ],
        }

    p = Planner(llm=fake_llm)
    assert p.backend == "llm"
    qtype, subs = p.plan("Who is older, Kurosawa or Spielberg?")
    assert qtype == "comparison"
    texts = [s.text for s in subs]
    assert texts.count("How old is Kurosawa?") == 1  # dedup worked
    assert len(subs) == 2


# ------------------------------------ end-to-end: coverage now fires
def test_pipeline_with_planner_flags_uncovered_bridge_hop():
    plan_pipe = Stage1Pipeline(planner=Planner())
    plan_pipe.index_documents(load_corpus())
    verifier = Verifier(EntailmentModel(prefer_fallback=True))

    state = plan_pipe.run("What nationality is the director of the film Ran?")
    # Planner exposed two hops.
    assert state.question_type == "bridge"
    assert len(state.sub_queries) == 2

    cov = verifier.check_coverage(state.sub_queries, state.evidence)
    # The second (nationality) hop is not covered by the single-pass evidence,
    # so coverage flags it -- the signal Stage 1 could not produce.
    assert "hop_1" in cov.detail["failing_hops"]

    # And the aggregate diagnosis is no longer a blind 'pass'.
    res = verifier.verify(state)
    assert res.diagnosis in ("planning_fault", "retrieval_fault")
    assert "hop_1" in res.failing_hops


def test_baseline_pipeline_unchanged_without_planner():
    # No planner -> single hop, exactly the Stage-1 behavior.
    base = Stage1Pipeline()
    base.index_documents(load_corpus())
    state = base.run("What nationality is the director of the film Ran?")
    assert len(state.sub_queries) == 1
    assert state.question_type == "single-hop"

