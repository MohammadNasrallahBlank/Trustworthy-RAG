"""Stage 6: run the full accuracy-vs-cost comparison and write a report.

    python examples/run_eval.py

Builds every config over the sample corpus, evaluates on the bundled eval set
(answerable multi-hop / single-hop + an unanswerable subset), calibrates the
full system's abstention thresholds, prints the comparison table, and writes
eval_report.md.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import Controller, Planner, Stage1Pipeline, Verifier  # noqa: E402
from srag import calibrate_thresholds, collect_points  # noqa: E402
from srag.evaluation import (  # noqa: E402
    CONFIGS, build_config, evaluate_config, compare_configs, render_markdown_report,
    load_eval_set,
)


def _load(name):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "data", name)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    documents = _load("sample_corpus.jsonl")
    eval_set = _load("eval_set.jsonl")
    dev_set = _load("dev_set.jsonl")

    # Calibrate the full system's abstention thresholds on the dev set.
    pipe = Stage1Pipeline(planner=Planner()).index_documents(documents)
    base_ctrl = Controller(pipe, Verifier(), tau=0.55, max_corrections=3)
    cal = calibrate_thresholds(collect_points(base_ctrl, dev_set))
    thresholds = (cal.tau_answer, cal.tau_abstain)
    print("Calibrated thresholds:", cal.to_dict())

    reports = []
    for name in CONFIGS:
        th = thresholds if name.startswith("full") else None
        runner = build_config(name, documents, thresholds=th)
        report = evaluate_config(runner, eval_set, seeds=1)
        reports.append(report)
        d = report.to_dict()
        print(f"  {name:20s} F1={d['F1'][0]:.3f} faithful={d['faithfulness'][0]:.3f} "
              f"abstP/R={d['abstention_P']:.2f}/{d['abstention_R']:.2f} ops={d['ops']:.1f}")

    comparison = compare_configs(reports)
    md = render_markdown_report(reports, comparison)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "eval_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print("\nWrote", os.path.relpath(out))
    if comparison.get("significance"):
        print("Verdict:", comparison["significance"]["verdict"])


if __name__ == "__main__":
    main()
