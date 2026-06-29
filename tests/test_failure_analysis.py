"""Failure auto-tagging tests (plan v2 Task 8)."""

from __future__ import annotations

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))

from failure_analysis import tag_failure, analyze_failures, FAILURE_TAGS  # noqa: E402


def _ev(*texts):
    return [{"id": f"d{i}", "text": t} for i, t in enumerate(texts)]


def test_uca_with_answer_absent_is_prior_knowledge():
    rec = {"category": "uca", "answerable": False, "abstained": False,
           "answers": [], "prediction": "Georges Bizet", "confidence": 0.7,
           "evidence": _ev("Nimbus syncs notes across devices.")}
    assert tag_failure(rec) == "answered_from_prior_knowledge"


def test_uca_high_confidence_is_verifier_too_lenient():
    rec = {"category": "uca", "answerable": False, "abstained": False,
           "answers": [], "prediction": "X", "confidence": 0.95,
           "evidence": _ev("totally unrelated text")}
    assert tag_failure(rec, lenient_conf=0.9) == "verifier_too_lenient"


def test_wrong_with_gold_absent_is_retrieval_missed():
    rec = {"category": "wrong", "answerable": True, "abstained": False,
           "answers": ["Paris"], "prediction": "Lyon", "confidence": 0.5,
           "evidence": _ev("Lyon is a city on the Rhone.")}
    assert tag_failure(rec) == "retrieval_missed_evidence"


def test_wrong_with_gold_present_is_evidence_ignored():
    rec = {"category": "wrong", "answerable": True, "abstained": False,
           "answers": ["Paris"], "prediction": "Lyon", "confidence": 0.5,
           "evidence": _ev("The capital of France is Paris.")}
    assert tag_failure(rec) == "evidence_ignored_by_generator"


def test_missed_with_gold_present_is_verifier_too_strict():
    rec = {"category": "missed", "answerable": True, "abstained": True,
           "answers": ["Paris"], "prediction": "", "confidence": 0.2,
           "evidence": _ev("The capital of France is Paris.")}
    assert tag_failure(rec) == "verifier_too_strict"


def test_correction_then_wrong_is_correction_irrelevant():
    rec = {"category": "wrong", "answerable": True, "abstained": False,
           "answers": ["Paris"], "prediction": "Berlin", "confidence": 0.5,
           "corrections": 2, "evidence": _ev("Berlin is in Germany.")}
    # corrections fired but gold still absent -> correction retrieved irrelevant
    assert tag_failure(rec) == "correction_retrieved_irrelevant"


def test_analyze_failures_counts_and_writes(tmp_path):
    records = [
        {"category": "correct", "answerable": True, "abstained": False,
         "answers": ["Paris"], "prediction": "Paris", "confidence": 0.8,
         "evidence": _ev("Paris is the capital.")},
        {"category": "uca", "answerable": False, "abstained": False,
         "answers": [], "prediction": "Bizet", "confidence": 0.7,
         "evidence": _ev("unrelated")},
        {"category": "wrong", "answerable": True, "abstained": False,
         "answers": ["Paris"], "prediction": "Lyon", "confidence": 0.5,
         "evidence": _ev("Lyon is on the Rhone.")},
    ]
    out = os.path.join(tmp_path, "failures.md")
    summary = analyze_failures(records, out_md=out)
    # correct rows are not failures
    assert summary["n_failures"] == 2
    assert set(summary["counts"]) <= set(FAILURE_TAGS)
    assert sum(summary["counts"].values()) == 2
    assert os.path.exists(out)
