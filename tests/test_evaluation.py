"""Tests for the Stage 6 evaluation harness (doc section 7)."""

from __future__ import annotations

import json
import os

import pytest

from srag.evaluation import (
    CONFIGS,
    build_config,
    compare_configs,
    evaluate_config,
    exact_match,
    load_eval_set,
    load_hotpot_style,
    normalize_answer,
    paired_bootstrap_pvalue,
    reciprocal_rank,
    render_markdown_report,
    retrieval_recall_at_k,
    token_f1,
)
from srag.evaluation.metrics import bootstrap_ci

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")
EVALSET = os.path.join(HERE, os.pardir, "data", "eval_set.jsonl")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------- metrics
def test_normalize_and_em():
    assert normalize_answer("The Japanese!") == "japanese"
    assert exact_match("Japanese", ["the japanese"]) == 1.0
    assert exact_match("French", ["Japanese"]) == 0.0


def test_token_f1_partial_credit():
    assert token_f1("Akira Kurosawa", ["Akira Kurosawa"]) == 1.0
    f1 = token_f1("Akira Kurosawa", ["Kurosawa"])
    assert 0.0 < f1 < 1.0


def test_retrieval_recall_and_mrr():
    retrieved = ["c3", "c1", "c2"]
    assert retrieval_recall_at_k(retrieved, ["c1", "c2"]) == 1.0
    assert retrieval_recall_at_k(retrieved, ["c1", "x9"]) == 0.5
    assert reciprocal_rank(retrieved, ["c1"]) == pytest.approx(1 / 2)
    assert reciprocal_rank(retrieved, ["zzz"]) == 0.0


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals, n_boot=500, seed=7)
    b = bootstrap_ci(vals, n_boot=500, seed=7)
    assert a == b                                  # deterministic with seed
    mean, lo, hi = a
    assert lo <= mean <= hi


def test_paired_bootstrap_detects_clear_win_and_tie():
    system = [1.0] * 10
    baseline = [0.0] * 10
    assert paired_bootstrap_pvalue(system, baseline, n_boot=500) < 0.05
    tie = paired_bootstrap_pvalue([0.5] * 10, [0.5] * 10, n_boot=500)
    assert tie >= 0.5


# ------------------------------------------------- datasets
def test_load_eval_set_and_hotpot_adapter():
    ds = load_eval_set(EVALSET)
    assert len(ds) == 10
    assert any(not d["answerable"] for d in ds)    # has an unanswerable subset
    adapted = load_hotpot_style([
        {"question": "Q1?", "answer": "Tokyo", "type": "bridge"},
        {"question": "Q2?", "answer": ""},          # -> unanswerable
    ])
    assert adapted[0]["answerable"] is True and adapted[0]["answers"] == ["Tokyo"]
    assert adapted[1]["answerable"] is False and adapted[1]["answers"] == []


# ------------------------------------------------- harness end-to-end
@pytest.fixture(scope="module")
def reports():
    docs = _load(CORPUS)
    evalset = _load(EVALSET)
    out = []
    for name in CONFIGS:
        runner = build_config(name, docs)
        out.append(evaluate_config(runner, evalset, seeds=1, n_boot=200))
    return {r.name: r for r in out}


def test_all_configs_run_over_full_eval_set(reports):
    assert set(reports) == set(CONFIGS)
    for r in reports.values():
        assert r.n == 10


def test_full_system_beats_baseline_on_f1(reports):
    # The headline: targeted correction lifts F1 above the reranked baseline.
    assert reports["full"].f1[0] > reports["reranker_baseline"].f1[0]


def test_planning_ablation_explains_the_gain(reports):
    # Removing the planner removes the bridge fix -> F1 falls back to baseline.
    assert reports["full_no_planning"].f1[0] <= reports["full"].f1[0]


def test_full_does_not_penalize_correct_abstention(reports):
    # On the unanswerable subset the full system should abstain (high recall).
    assert reports["full"].abstention_recall >= 0.75


def test_compare_and_report_render(reports):
    rs = list(reports.values())
    comp = compare_configs(rs)
    assert comp["significance"] is not None
    assert "verdict" in comp["significance"]
    md = render_markdown_report(rs, comp)
    assert "evaluation report" in md.lower()
    assert "reranker_baseline" in md and "full" in md
    assert "Accuracy vs cost" in md
