"""Tests for the HotpotQA distractor loader + harness integration."""

from __future__ import annotations

import json
import os

from srag.evaluation import build_config, evaluate_config, load_hotpot_distractor

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, os.pardir, "data", "hotpot_sample.json")


def _records():
    with open(SAMPLE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_loader_builds_corpus_and_maps_gold():
    docs, ds = load_hotpot_distractor(_records())
    assert len(ds) == 4
    # Corpus includes distractor paragraphs, not just gold.
    assert len(docs) >= 6
    assert any(d['id'] == 'citizen-kane' for d in docs)  # a distractor
    # Gold supporting facts map onto indexed chunk ids.
    q1 = next(d for d in ds if "Ran" in d["question"])
    assert q1["supporting_chunk_ids"] == ["ran-film", "akira-kurosawa"]
    assert all(any(c["id"] == gid for c in docs) for gid in q1["supporting_chunk_ids"])


def test_loader_marks_unanswerable():
    recs = _records() + [{"_id": "u1", "question": "Unknown?", "answer": "",
                          "supporting_facts": [], "context": []}]
    _, ds = load_hotpot_distractor(recs)
    assert ds[-1]["answerable"] is False and ds[-1]["answers"] == []


def test_harness_runs_on_hotpot_and_full_beats_baseline():
    docs, ds = load_hotpot_distractor(_records())
    rep = {}
    for name in ["naive", "reranker_baseline", "full"]:
        runner = build_config(name, docs, pre_chunked=True)
        rep[name] = evaluate_config(runner, ds, seeds=1, n_boot=200)
    # Real retrieval metrics are populated in the distractor setting.
    assert rep["reranker_baseline"].recall_at_k[0] > 0.0
    assert rep["reranker_baseline"].mrr[0] > 0.0
    # The full loop is at least as accurate as the reranked baseline here.
    assert rep["full"].f1[0] >= rep["reranker_baseline"].f1[0]
