"""Coverage / UCA operating curves (plan v2 section 3.7).

A thresholded system (retrieval_score_abstain / guarded / guarded+correct) does
not have a single coverage -- it has a *curve*: sweep the abstention threshold
and trade answerable coverage against the Unsupported-Confident-Answer rate. We
build that curve, pick a default operating point ("answer >= X% of answerable
while minimizing UCA"), and read UCA off the curve at any matched coverage so
systems are compared fairly (section 3.5).

A question is "answered" at threshold tau iff its confidence >= tau:
  coverage(tau) = answered answerable / total answerable
  uca(tau)      = answered unanswerable / total unanswerable
"""

from __future__ import annotations

from typing import Optional, Sequence


def coverage_uca_curve(records: Sequence[dict]) -> list[dict]:
    """Sweep the abstention threshold over confidence scores.

    `records`: per-question dicts with `confidence` (float) and `answerable`
    (bool). Returns a list of {threshold, coverage, uca}, sorted by coverage
    ascending. Includes the answer-nothing point (coverage 0) and the
    answer-everything point (coverage 1).
    """
    answerable = [float(r["confidence"]) for r in records if r["answerable"]]
    unanswerable = [float(r["confidence"]) for r in records if not r["answerable"]]
    n_ans = len(answerable)
    n_unans = len(unanswerable)

    # Candidate thresholds: each observed score (answer all with score >= tau),
    # plus +inf (answer nothing).
    scores = sorted({float(r["confidence"]) for r in records})
    thresholds = scores + [float("inf")]

    points: list[dict] = []
    for tau in thresholds:
        cov = (sum(1 for c in answerable if c >= tau) / n_ans) if n_ans else 0.0
        uca = (sum(1 for c in unanswerable if c >= tau) / n_unans) if n_unans else 0.0
        points.append({"threshold": tau, "coverage": cov, "uca": uca})

    points.sort(key=lambda p: (p["coverage"], p["uca"]))
    return points


def pick_operating_point(curve: Sequence[dict], *,
                         min_coverage: float = 0.8) -> Optional[dict]:
    """Pick the threshold minimizing UCA subject to coverage >= min_coverage.

    Ties on UCA break toward higher coverage. Returns None if no point reaches
    `min_coverage`.
    """
    feasible = [p for p in curve if p["coverage"] >= min_coverage - 1e-9]
    if not feasible:
        return None
    return min(feasible, key=lambda p: (p["uca"], -p["coverage"]))


def uca_at_coverage(curve: Sequence[dict], target_coverage: float) -> float:
    """Lowest UCA achievable while still covering >= `target_coverage`.

    This is the value used for "UCA at matched coverage" comparisons (section
    3.5): read each thresholded system off its curve at a shared coverage.
    Returns 1.0 if the target coverage is unreachable (worst case).
    """
    op = pick_operating_point(curve, min_coverage=target_coverage)
    return op["uca"] if op is not None else 1.0


def plot_curves(systems: dict, out_png: str, *, title: str = "Coverage vs UCA") -> Optional[str]:
    """Plot coverage (x) vs UCA (y) for each system.

    `systems`: name -> either a curve (list of {coverage, uca}) for thresholded
    systems, or a single point dict {coverage, uca} for fixed systems
    (plain_rag_always_answer, prompted_abstain_rag). Returns the path written, or
    None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, data in systems.items():
        if isinstance(data, dict):  # single point
            ax.scatter([data["coverage"]], [data["uca"]], s=70, label=name, zorder=5)
        else:
            pts = sorted(data, key=lambda p: p["coverage"])
            ax.plot([p["coverage"] for p in pts], [p["uca"] for p in pts],
                    marker="o", markersize=3, label=name)
    ax.set_xlabel("Answerable coverage")
    ax.set_ylabel("UCA rate (answered unanswerables)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
