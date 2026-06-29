"""Comparison report + accuracy-vs-cost curve (doc section 7).

The headline result is a curve, not a point: each config contributes a point of
(cost, accuracy/faithfulness). The report also runs the key significance test --
full system vs. the reranked baseline -- via a paired bootstrap over per-question
F1. If the loop can't beat the reranked baseline within CIs, that is reported
plainly as a (legitimate) negative result, per the doc.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .harness import ConfigReport
from .metrics import paired_bootstrap_pvalue

_BASELINE = "reranker_baseline"
_FULL = "full"


def _aligned_f1(report: ConfigReport) -> list[float]:
    return [r.f1 for r in report.records if r.answerable and r.f1 is not None]


def compare_configs(reports: Sequence[ConfigReport], *,
                    baseline: str = _BASELINE, system: str = _FULL,
                    seed: int = 0) -> dict:
    """Build the comparison: per-config rows + the full-vs-baseline test."""
    by_name = {r.name: r for r in reports}
    out = {"configs": [r.to_dict() for r in reports], "significance": None}

    if baseline in by_name and system in by_name:
        b, s = by_name[baseline], by_name[system]
        # Align per-question F1 (answerable only); requires identical datasets.
        bf = _aligned_f1(b)
        sf = _aligned_f1(s)
        if len(bf) == len(sf) and bf:
            p = paired_bootstrap_pvalue(sf, bf, seed=seed)
            delta = (sum(sf) / len(sf)) - (sum(bf) / len(bf))
            out["significance"] = {
                "comparison": f"{system} vs {baseline}",
                "metric": "answer F1 (answerable items)",
                "delta_mean_f1": round(delta, 4),
                "p_value_one_sided": round(p, 4),
                "verdict": _verdict(delta, p),
            }
    return out


def _verdict(delta: float, p: float) -> str:
    if delta > 0 and p < 0.05:
        return "full system beats the reranked baseline (significant)"
    if delta > 0:
        return "full system ahead but NOT significant within CIs (treat as inconclusive)"
    if delta == 0:
        return "no difference vs the reranked baseline"
    return "full system does NOT beat the reranked baseline (negative result -- report it)"


def render_markdown_report(reports: Sequence[ConfigReport],
                           comparison: Optional[dict] = None) -> str:
    comparison = comparison or compare_configs(reports)
    lines: list[str] = []
    lines.append("# Self-Correcting RAG — evaluation report\n")
    lines.append(
        "Metrics reported separately; CIs are 95% percentile bootstraps. "
        "Accuracy-vs-cost is the headline: read F1/faithfulness against ops & "
        "latency. The bar to beat is `reranker_baseline`.\n"
    )

    # Main table.
    header = (
        "| config | EM | F1 | faithful | recall@k | MRR | abst P/R | "
        "ops | latency ms | corr |"
    )
    sep = "|" + "---|" * 11
    lines.append(header)
    lines.append(sep)
    for r in reports:
        d = r.to_dict()
        lines.append(
            f"| {d['config']} "
            f"| {d['EM'][0]:.3f} "
            f"| {d['F1'][0]:.3f} [{d['F1'][1]:.2f},{d['F1'][2]:.2f}] "
            f"| {d['faithfulness'][0]:.3f} "
            f"| {d['recall@k'][0]:.3f} "
            f"| {d['MRR'][0]:.3f} "
            f"| {d['abstention_P']:.2f}/{d['abstention_R']:.2f} "
            f"| {d['ops']:.1f} "
            f"| {d['latency_ms']:.2f} "
            f"| {d['corrections']:.2f} |"
        )

    # Accuracy-vs-cost curve (ops as the cost axis).
    lines.append("\n## Accuracy vs cost (F1 vs operations)\n")
    pts = sorted(((r.mean_operations, r.f1[0], r.name) for r in reports))
    lines.append("| config | ops (cost) | F1 |")
    lines.append("|---|---|---|")
    for ops, f1, name in pts:
        lines.append(f"| {name} | {ops:.1f} | {f1:.3f} |")

    # Significance.
    sig = comparison.get("significance")
    lines.append("\n## Key comparison\n")
    if sig:
        lines.append(
            f"**{sig['comparison']}** on {sig['metric']}: "
            f"Δmean F1 = {sig['delta_mean_f1']:+.3f}, "
            f"one-sided bootstrap p = {sig['p_value_one_sided']:.3f}.\n"
        )
        lines.append(f"> {sig['verdict']}\n")
    else:
        lines.append("_Not enough aligned data to run the paired test._\n")

    lines.append(
        "\n_Note: with the deterministic offline stack, seeds are degenerate and "
        "the corpus is tiny, so absolute numbers are illustrative. The harness "
        "supports stochastic generators and real datasets (HotpotQA / "
        "2WikiMultiHopQA / MuSiQue via `load_hotpot_style`); the doc calls for "
        "≥ a few hundred questions and ≥3 seeds for publishable numbers._\n"
    )
    return "\n".join(lines)
