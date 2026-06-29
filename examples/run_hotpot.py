"""Run the harness on HotpotQA (distractor setting).

    python examples/run_hotpot.py                       # bundled 4-record sample
    python examples/run_hotpot.py /path/hotpot_dev_distractor_v1.json [N]

Loads HotpotQA-format records, builds the corpus from each question's context
(gold + distractor paragraphs — the real distractor setting, no leakage), maps
supporting_facts to gold chunk ids, runs the config curve, and writes
hotpot_report.md. The same code runs on the full dev set locally.

Note: the offline extractive generator handles extractive bridge answers but not
synthesized ones (e.g. yes/no comparisons), so plug in a real LLM generator
(`srag.llm`) for publishable numbers.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag.evaluation import (  # noqa: E402
    build_config, evaluate_config, compare_configs, render_markdown_report,
    load_hotpot_distractor,
)

CONFIGS_TO_RUN = ["naive", "reranker_baseline", "planning", "full"]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, os.pardir, "data", "hotpot_sample.json")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit:
        records = records[:limit]

    documents, dataset = load_hotpot_distractor(records)
    print(f"Loaded {len(records)} HotpotQA records -> {len(documents)} chunks, "
          f"{len(dataset)} questions (distractor setting)")

    reports = []
    for name in CONFIGS_TO_RUN:
        runner = build_config(name, documents, pre_chunked=True)
        report = evaluate_config(runner, dataset, seeds=1, n_boot=1000)
        reports.append(report)
        d = report.to_dict()
        print(f"  {name:18s} EM={d['EM'][0]:.3f} F1={d['F1'][0]:.3f} "
              f"recall@k={d['recall@k'][0]:.3f} MRR={d['MRR'][0]:.3f} ops={d['ops']:.1f}")

    comparison = compare_configs(reports)
    md = render_markdown_report(reports, comparison)
    out = os.path.join(here, os.pardir, "hotpot_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print("\nWrote", os.path.relpath(out))
    if comparison.get("significance"):
        print("Verdict:", comparison["significance"]["verdict"])


if __name__ == "__main__":
    main()
