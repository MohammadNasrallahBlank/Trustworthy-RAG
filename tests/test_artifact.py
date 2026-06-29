"""Reproducibility artifact round-trip (plan v2 §3.8, Task 5)."""

from __future__ import annotations

import json
import os

from srag.evaluation.artifact import RunArtifact, REQUIRED_META_KEYS


def _meta():
    return {
        "dataset_version": "hotpotqa-sample-v1",
        "model_name": "offline-extractive",
        "prompt_template": "grounded-v1",
        "retriever_config": {"fusion": "rrf", "k": 50},
        "reranker_config": {"backend": "offline"},
        "verifier_thresholds": {"tau_answer": 0.55, "tau_abstain": 0.30},
        "seed": 0,
    }


def _examples():
    return [
        {"question": "Where is the Eiffel Tower?", "gold": ["Paris"],
         "evidence": [{"id": "d0", "text": "The Eiffel Tower is in Paris."}],
         "answer": "Paris", "abstained": False, "category": "correct",
         "confidence": 0.82},
        {"question": "Who composed Carmen?", "gold": [],
         "evidence": [], "answer": "", "abstained": True, "category": "refusal",
         "confidence": 0.1},
    ]


def test_artifact_round_trips_and_has_required_keys(tmp_path):
    out = os.path.join(tmp_path, "run1")
    summary = {"uca_rate": 0.0, "coverage": 0.5,
               "cis": {"coverage": [0.5, 0.1, 0.9]}}
    RunArtifact.save(out, meta=_meta(), per_example=_examples(), summary=summary)

    assert os.path.exists(os.path.join(out, "manifest.json"))
    assert os.path.exists(os.path.join(out, "examples.jsonl"))
    assert os.path.exists(os.path.join(out, "report.md"))

    manifest = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    for key in REQUIRED_META_KEYS:
        assert key in manifest["meta"], key
    assert manifest["summary"]["coverage"] == 0.5

    loaded = RunArtifact.load(out)
    assert loaded.meta["model_name"] == "offline-extractive"
    assert len(loaded.per_example) == 2
    assert loaded.per_example[0]["question"] == "Where is the Eiffel Tower?"
    assert loaded.summary["uca_rate"] == 0.0


def test_artifact_rejects_incomplete_meta(tmp_path):
    out = os.path.join(tmp_path, "run2")
    bad = {"model_name": "x"}  # missing most required keys
    try:
        RunArtifact.save(out, meta=bad, per_example=[], summary={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for incomplete meta")
