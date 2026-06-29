"""Coverage / UCA operating-curve tests (plan v2 §3.7, Task 4)."""

from __future__ import annotations

from srag.evaluation.curves import (
    coverage_uca_curve, pick_operating_point, uca_at_coverage,
)


def _records():
    # 5 answerable, 2 unanswerable, with known confidence scores.
    ans = [0.9, 0.8, 0.7, 0.6, 0.5]
    unans = [0.95, 0.55]
    return (
        [{"confidence": c, "answerable": True} for c in ans]
        + [{"confidence": c, "answerable": False} for c in unans]
    )


def test_curve_is_sorted_by_coverage_and_bounded():
    curve = coverage_uca_curve(_records())
    covs = [p["coverage"] for p in curve]
    assert covs == sorted(covs)
    for p in curve:
        assert 0.0 <= p["coverage"] <= 1.0
        assert 0.0 <= p["uca"] <= 1.0
    # extremes present: answer-nothing (cov 0) and answer-all (cov 1).
    assert any(abs(p["coverage"] - 1.0) < 1e-9 for p in curve)
    assert any(abs(p["coverage"]) < 1e-9 for p in curve)


def test_pick_operating_point_minimizes_uca_subject_to_coverage():
    curve = coverage_uca_curve(_records())
    op = pick_operating_point(curve, min_coverage=0.8)
    assert op["coverage"] >= 0.8 - 1e-9
    # At coverage 0.8 the lowest achievable UCA is 0.5 (drop the unanswerable
    # scoring 0.55 by setting the threshold at 0.6).
    assert abs(op["uca"] - 0.5) < 1e-9
    assert abs(op["threshold"] - 0.6) < 1e-9


def test_pick_operating_point_returns_none_when_coverage_unreachable():
    curve = coverage_uca_curve(_records())
    assert pick_operating_point(curve, min_coverage=1.01) is None


def test_uca_at_coverage_matches_curve():
    curve = coverage_uca_curve(_records())
    # at coverage 0.8 the best (lowest) UCA is 0.5
    assert abs(uca_at_coverage(curve, 0.8) - 0.5) < 1e-9
