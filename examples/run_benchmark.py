"""Scale benchmark: the experiment that turns the paper's limitations into results.

    # offline smoke (no weights, runs anywhere — for wiring checks only):
    python examples/run_benchmark.py --backend offline --n 8

    # REAL run (needs `pip install -e .[models]` and a GPU/Colab):
    python examples/run_benchmark.py \
        --backend real --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset hotpot_dev_distractor_v1.json \
        --n 300 --unanswerable-frac 0.25 --seeds 3 --out-dir results

Loads HotpotQA (distractor schema), optionally holds out gold context for a
fraction of questions to build a genuine unanswerable subset, runs every config
(naive → reranker_baseline → planning → full + ablations) with REAL models,
computes bootstrap CIs and a paired significance test vs. the reranked baseline,
and writes results.json, report.md, and the accuracy-vs-cost figure.

This is the script to run for publishable numbers; `--backend offline` exists
only so the plumbing can be exercised without model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import EntailmentModel, calibrate_thresholds, collect_points  # noqa: E402
from srag.evaluation import (  # noqa: E402
    CONFIGS, build_config, evaluate_config, compare_configs, render_markdown_report,
    load_hotpot_distractor, make_unanswerable_holdout, sample_dataset,
    accuracy_cost_curve, diagnosis_routing_diagram,
)


def load_records(path, download):
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if download:
        # Real datasets (HotpotQA / 2Wiki) via the `datasets` library — Colab path.
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        recs = []
        for r in ds:
            ctx = list(zip(r["context"]["title"], r["context"]["sentences"]))
            sf = list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]))
            recs.append({"question": r["question"], "answer": r["answer"],
                         "type": r.get("type", "unknown"),
                         "supporting_facts": [list(x) for x in sf],
                         "context": [[t, list(s)] for t, s in ctx]})
        return recs
    # Fallback: the bundled sample.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, os.pardir, "data", "hotpot_sample.json")) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["offline", "real"], default="offline")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default=None, help="path to a HotpotQA-format JSON")
    ap.add_argument("--download", action="store_true", help="load HotpotQA via `datasets`")
    ap.add_argument("--n", type=int, default=None, help="sample this many questions")
    ap.add_argument("--unanswerable-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--dev-frac", type=float, default=0.2,
                    help="fraction held out to calibrate abstention thresholds")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--llm-planner", action="store_true",
                    help="use the LLM for query decomposition (else rule-based)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    records = load_records(args.dataset, args.download)
    documents, dataset = load_hotpot_distractor(records)
    documents, dataset = sample_dataset(documents, dataset, n=args.n, seed=0)
    if args.unanswerable_frac > 0:
        documents, dataset = make_unanswerable_holdout(
            documents, dataset, frac=args.unanswerable_frac, seed=0)
    n_ans = sum(1 for d in dataset if d["answerable"])
    print(f"Dataset: {len(dataset)} questions ({n_ans} answerable, "
          f"{len(dataset) - n_ans} unanswerable), {len(documents)} chunks")

    prefer_fallback = args.backend == "offline"
    chat = None
    if args.backend == "real":
        from srag import TransformersChat
        chat = TransformersChat(args.model, max_new_tokens=args.max_new_tokens)
        print(f"Real backend: {args.model} + sentence-transformers")
    entailment = EntailmentModel(prefer_fallback=prefer_fallback)

    # Dev/test split: calibrate abstention thresholds on dev, REPORT on test.
    n_dev = int(round(len(dataset) * args.dev_frac))
    dev, test = dataset[:n_dev], dataset[n_dev:]
    thresholds = None
    if dev:
        base = build_config("full", documents, pre_chunked=True,
                            prefer_fallback=prefer_fallback, chat=chat,
                            llm_planner=args.llm_planner)
        cal = calibrate_thresholds(collect_points(base, dev))
        thresholds = (cal.tau_answer, cal.tau_abstain)
        print(f"Calibrated on {len(dev)} dev: tau_answer={cal.tau_answer:.3f} "
              f"tau_abstain={cal.tau_abstain:.3f}; reporting on {len(test)} test")
    else:
        test = dataset

    reports = []
    for name in args.configs:
        th = thresholds if name.startswith("full") else None
        runner = build_config(name, documents, pre_chunked=True,
                              prefer_fallback=prefer_fallback, chat=chat,
                              thresholds=th, llm_planner=args.llm_planner)
        rep = evaluate_config(runner, test, entailment=entailment,
                              seeds=args.seeds, n_boot=2000)
        reports.append(rep)
        d = rep.to_dict()
        print(f"  {name:20s} EM={d['EM'][0]:.3f} F1={d['F1'][0]:.3f} "
              f"faithful={d['faithfulness'][0]:.3f} recall@k={d['recall@k'][0]:.3f} "
              f"abstP/R={d['abstention_P']:.2f}/{d['abstention_R']:.2f} ops={d['ops']:.1f}")

    os.makedirs(args.out_dir, exist_ok=True)
    comparison = compare_configs(reports)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"configs": [r.to_dict() for r in reports],
                   "comparison": comparison,
                   "meta": {"backend": args.backend, "model": args.model,
                            "n": len(test), "answerable": sum(1 for d in test if d["answerable"]),
                            "seeds": args.seeds}}, f, indent=2)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(render_markdown_report(reports, comparison))
    try:
        accuracy_cost_curve(reports, os.path.join(args.out_dir, "accuracy_cost.png"))
        diagnosis_routing_diagram(os.path.join(args.out_dir, "routing.png"))
    except Exception as e:
        print("Figure generation skipped:", e)

    print(f"\nWrote {args.out_dir}/results.json, report.md, accuracy_cost.png")
    if comparison.get("significance"):
        print("Verdict:", comparison["significance"]["verdict"])


if __name__ == "__main__":
    main()
