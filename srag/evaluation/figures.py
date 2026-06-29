"""Figure generation for the report (doc section 7's "headline is a curve").

`accuracy_cost_curve` renders the accuracy-vs-cost scatter that is the whole
point of evaluating self-correction: does the extra compute buy accuracy? It
plots each config as a point of (cost, F1) with the reranked baseline marked, so
the full loop has to sit up-and-to-the-right of it to be worth the spend.

Matplotlib only; no seaborn. Saves a PNG and returns its path.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .harness import ConfigReport


def accuracy_cost_curve(reports: Sequence[ConfigReport], out_png: str,
                        *, cost: str = "ops", metric: str = "f1",
                        title: str = "Accuracy vs cost") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _cost(r):
        return r.mean_operations if cost == "ops" else r.mean_latency_ms

    def _acc(r):
        return getattr(r, metric)[0] if metric in ("f1", "em") else r.f1[0]

    def _err(r):
        m, lo, hi = getattr(r, metric) if metric in ("f1", "em") else r.f1
        return [[m - lo], [hi - m]]

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in reports:
        x, y = _cost(r), _acc(r)
        is_full = r.name == "full"
        is_base = r.name == "reranker_baseline"
        color = "#2563eb" if is_full else ("#dc2626" if is_base else "#6b7280")
        marker = "*" if is_full else ("s" if is_base else "o")
        size = 320 if is_full else (150 if is_base else 90)
        ax.errorbar(x, y, yerr=_err(r), fmt="none", ecolor=color, alpha=0.5, capsize=3)
        ax.scatter([x], [y], s=size, c=color, marker=marker, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(r.name, (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color="#111827")

    ax.set_xlabel("Cost  (mean operations per query)" if cost == "ops"
                  else "Cost  (mean latency, ms)", fontsize=11)
    ax.set_ylabel(f"Answer {metric.upper()}  (95% bootstrap CI)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def diagnosis_routing_diagram(out_png: str) -> str:
    """A static diagram of the diagnosis -> targeted-correction routing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    linewidth=1.2, edgecolor="#111827", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
                color="#111827", wrap=True)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.1, color="#374151"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.015, label, ha="center",
                    fontsize=8, color="#6b7280")

    box(0.36, 0.86, 0.28, 0.10, "Verifier\n(confidence + diagnosis)", "#dbeafe")
    routes = [
        ("retrieval_fault", "Re-retrieve failing hop\nreformulate → HyDE → web", "#fee2e2", 0.02),
        ("planning_fault", "Fill templated hop from\nprerequisite, re-retrieve", "#fef3c7", 0.27),
        ("generation_fault", "Regenerate on relevant\nchunks (less distraction)", "#dcfce7", 0.52),
        ("conflict", "Resolve by recency /\nauthority / majority", "#ede9fe", 0.77),
    ]
    for label, action, color, x in routes:
        ax.text(x + 0.11, 0.66, label, ha="center", fontsize=8.5, fontweight="bold", color="#374151")
        box(x, 0.40, 0.22, 0.20, action, color)
        arrow(0.50, 0.86, x + 0.11, 0.605, "")
        arrow(x + 0.11, 0.40, x + 0.11, 0.27, "")
    box(0.30, 0.11, 0.40, 0.14, "Bounded loop\nstop on: confidence ≥ τ  |  budget  |  no new evidence\n→ answer  or  calibrated abstention", "#f3f4f6")
    ax.text(0.5, 0.985, "Diagnose, then target the fix", ha="center", fontsize=13,
            fontweight="bold", color="#111827")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png
