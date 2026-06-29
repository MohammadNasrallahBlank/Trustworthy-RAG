"""Turn a benchmark matrix into a results section + figure for the paper/post.

    python tools/ingest_results.py results/matrix

Reads the per-cell JSONs written by examples/run_matrix.py, builds a clean
results table (RESULTS.md), an accuracy-vs-cost figure per (dataset, model), and
a machine-readable results/REAL_RESULTS.json the paper/README can cite.
"""
import json
import os
import sys


def load_cells(matrix_dir):
    cells = []
    for fn in sorted(os.listdir(matrix_dir)):
        if fn.endswith(".json") and "__" in fn and fn != "summary.json":
            cells.append(json.load(open(os.path.join(matrix_dir, fn))))
    return cells


def table_md(cells):
    rows = ["| dataset | model | config | EM | F1 [95% CI] | faithful | recall@k | MRR | abst P/R | ops |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["dataset"], c["model"], c["config"])):
        m = c["metrics"]
        rows.append(
            f"| {c['dataset']} | {c['model'].split('/')[-1]} | {c['config']} "
            f"| {m['EM'][0]:.3f} | {m['F1'][0]:.3f} [{m['F1'][1]:.2f},{m['F1'][2]:.2f}] "
            f"| {m['faithfulness'][0]:.3f} | {m['recall@k'][0]:.3f} | {m['MRR'][0]:.3f} "
            f"| {m['abstention_P']:.2f}/{m['abstention_R']:.2f} | {m['ops']:.1f} |")
    return "\n".join(rows)


def significance(cells):
    from srag.evaluation.metrics import paired_bootstrap_pvalue  # noqa
    by = {}
    for c in cells:
        by.setdefault((c["dataset"], c["model"]), {})[c["config"]] = c
    out = {}
    for (ds, model), confs in by.items():
        if "full" in confs and "reranker_baseline" in confs:
            sf = confs["full"].get("f1_per_q", [])
            bf = confs["reranker_baseline"].get("f1_per_q", [])
            if sf and len(sf) == len(bf):
                p = paired_bootstrap_pvalue(sf, bf, seed=0)
                delta = sum(sf) / len(sf) - sum(bf) / len(bf)
                out[f"{ds} | {model.split('/')[-1]}"] = {
                    "delta_f1": round(delta, 4), "p_value": round(p, 4)}
    return out


def figures(cells, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for c in cells:
        groups.setdefault((c["dataset"], c["model"]), []).append(c)
    paths = []
    for (ds, model), cs in groups.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in cs:
            m = c["metrics"]
            x, y = m["ops"], m["F1"][0]
            full = c["config"] == "full"
            base = c["config"] == "reranker_baseline"
            color = "#2563eb" if full else ("#dc2626" if base else "#6b7280")
            ax.errorbar(x, y, yerr=[[y - m["F1"][1]], [m["F1"][2] - y]], fmt="none",
                        ecolor=color, alpha=0.5, capsize=3)
            ax.scatter([x], [y], s=320 if full else 120, c=color,
                       marker="*" if full else ("s" if base else "o"),
                       edgecolors="white", zorder=3)
            ax.annotate(c["config"], (x, y), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)
        ax.set_xlabel("Cost (mean operations / query)")
        ax.set_ylabel("Answer F1 (95% CI)")
        ax.set_title(f"{ds} · {model.split('/')[-1]}", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.margins(0.15)
        fig.tight_layout()
        p = os.path.join(out_dir, f"curve_{ds}_{model.split('/')[-1]}.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    matrix_dir = sys.argv[1] if len(sys.argv) > 1 else "results/matrix"
    cells = load_cells(matrix_dir)
    if not cells:
        print("no cells in", matrix_dir)
        return
    sig = significance(cells)
    md = ["# Real benchmark results\n",
          f"Real open models on real datasets. {len(cells)} cells.\n",
          table_md(cells),
          "\n## Full vs reranked baseline (paired bootstrap)\n",
          "| dataset · model | Δ F1 | p (one-sided) | verdict |",
          "|---|---|---|---|"]
    for k, v in sig.items():
        verdict = ("**beats baseline (sig.)**" if v["delta_f1"] > 0 and v["p_value"] < 0.05
                   else ("ahead, not significant" if v["delta_f1"] > 0 else "no gain"))
        md.append(f"| {k} | {v['delta_f1']:+.3f} | {v['p_value']:.3f} | {verdict} |")
    figs = figures(cells, matrix_dir)
    md.append("\n## Figures\n")
    for p in figs:
        md.append(f"- `{os.path.relpath(p)}`")
    out_md = os.path.join(matrix_dir, "RESULTS.md")
    open(out_md, "w").write("\n".join(md) + "\n")
    json.dump({"cells": cells, "significance": sig},
              open(os.path.join(matrix_dir, "REAL_RESULTS.json"), "w"), indent=2)
    print("wrote", out_md, "and", len(figs), "figures")


if __name__ == "__main__":
    main()
