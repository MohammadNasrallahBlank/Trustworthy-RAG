"""Tests for the real open-model adapters (offline, via a fake chat callable)."""

from __future__ import annotations

from srag import make_llm_generator_fn, make_llm_planner_fn
from srag.generator import GroundedGenerator
from srag.llm import _loads_json
from srag.planner import Planner
from srag.state import Chunk


def test_generator_adapter_coerces_fenced_json_and_filters_citations():
    def fake_chat(prompt):
        return ('```json\n{"answer":"Japanese",'
                '"claims":[{"text":"Kurosawa was Japanese","citations":["c1"]},'
                '{"text":"hallucination","citations":["nope"]}],'
                '"unsupported_facts":[],"answerable":true}\n```')

    gen = GroundedGenerator(llm=make_llm_generator_fn(fake_chat))
    res = gen.generate("What nationality?", [Chunk(id="c1", text="Kurosawa was Japanese.")])
    assert res.answer == "Japanese"
    cites = [cid for c in res.claims for cid in c.citations]
    assert "c1" in cites and "nope" not in cites      # invalid citation dropped


def test_planner_adapter_builds_templated_bridge_hop():
    def fake_chat(prompt):
        return ('{"type":"bridge","sub_queries":['
                '{"text":"Who directed Ran?"},'
                '{"text":"What nationality is the director?",'
                '"depends_on":"hop_0","template":"What nationality is {entity}?"}]}')

    planner = Planner(llm=make_llm_planner_fn(fake_chat))
    qtype, subs = planner.plan("What nationality is the director of Ran?")
    assert qtype == "bridge"
    assert len(subs) == 2
    assert subs[1].depends_on == "hop_0"
    assert subs[1].resolved is False
    assert subs[1].fill("Akira Kurosawa").text == "What nationality is Akira Kurosawa?"


def test_loads_json_is_robust_to_junk():
    assert _loads_json("garbage") == {}
    assert _loads_json('prefix {"a": 1} suffix')["a"] == 1
    assert _loads_json('```json\n{"x": true}\n```')["x"] is True
