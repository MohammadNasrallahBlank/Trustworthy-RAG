"""Offline smoke for the trust runner (plan v2 Task 6 / 6b)."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "examples"))

import run_trust  # noqa: E402


def test_run_trust_docs_offline_end_to_end(tmp_path):
    out = os.path.join(tmp_path, "trust_docs")
    result = run_trust.run_trust("docs", backend="offline", out_dir=out,
                                 seed=0, verbose=False)
    rows = result["summary"]
    systems = {r["system"] for r in rows}
    assert systems == set(run_trust.TRUST_CONFIGS)
    # Each system wrote a reproducibility artifact (manifest + examples + report).
    for name in run_trust.TRUST_CONFIGS:
        d = os.path.join(out, name.replace("+", "_"))
        assert os.path.exists(os.path.join(d, "manifest.json"))
        assert os.path.exists(os.path.join(d, "examples.jsonl"))
        assert os.path.exists(os.path.join(d, "report.md"))
    assert os.path.exists(os.path.join(out, "summary.json"))


def test_prepare_data_docs_leakage_checked():
    docs, data, prep = run_trust.prepare_data("docs")
    assert len(docs) >= 15
    assert any(d["answerable"] for d in data)
    assert any(not d["answerable"] for d in data)
    assert "dropped_unanswerable" in prep


def test_mcnemar_p_matches_known_values():
    # All discordant pairs one direction -> very small p.
    assert run_trust._mcnemar_p(20, 0) < 1e-4
    # Symmetric -> p == 1.0
    assert run_trust._mcnemar_p(5, 5) == 1.0
    assert run_trust._mcnemar_p(0, 0) == 1.0


def test_paired_uca_significance_flags_guarded_beating_plain():
    # Build fake reports: plain answers all unanswerables; guarded abstains on most.
    class _Rep:
        def __init__(self, records):
            self.records = records

    qs = [f"u{i}" for i in range(44)]
    plain = _Rep([{"question": q, "answerable": False, "abstained": False} for q in qs])
    # guarded abstains on 20 of 44 (answers 24 -> UCA)
    guarded = _Rep([{"question": q, "answerable": False, "abstained": (i < 20)}
                    for i, q in enumerate(qs)])
    reports = {"plain_rag_always_answer": plain, "guarded": guarded}
    sig = run_trust._paired_uca_significance(reports)
    key = "guarded_vs_plain_rag_always_answer"
    assert key in sig
    assert sig[key]["delta_uca"] < 0          # guarded has fewer UCAs
    assert sig[key]["mcnemar_p"] < 0.05
    assert sig[key]["bootstrap_p"] < 0.05
