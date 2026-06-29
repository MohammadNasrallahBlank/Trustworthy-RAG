"""Resumable benchmark MATRIX: datasets x models x configs (real models).

Designed for a long unattended run on a single GPU, monitorable via the bridge.
Every (dataset, model, config) cell is checkpointed to its own JSON, so the run
resumes after any interruption and writes a running summary after each cell.

    python examples/run_matrix.py --backend real \
        --datasets hotpotqa,2wiki,musique \
        --models Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct \
        --configs naive,reranker_baseline,full \
        --n 300 --unanswerable-frac 0.25 --dev-frac 0.2 \
        --out-dir results/matrix

Put dataset files in --data-dir (default data/raw):
  hotpotqa -> hotpotqa_dev.json  (HotpotQA distractor schema; `tools/get_datasets.py`)
  2wiki    -> 2wiki_dev.json     (same schema)
  musique  -> musique_dev.jsonl  (MuSiQue -Ans/-Full)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import EntailmentModel, calibrate_thresholds, collect_points  # noqa: E402
from srag.evaluation import (  # noqa: E402
    build_config, evaluate_config, compare_configs, render_markdown_report,
    accuracy_cost_curve, load_hotpot_distractor, load_musique,
    make_unanswerable_holdout, sample_dataset,
)
from srag.evaluation.metrics import paired_bootstrap_pvalue  # noqa: E402

LOADERS = {"hotpotqa": ("hotpotqa_dev.json", load_hotpot_distractor),
           "2wiki": ("2wiki_dev.json", load_hotpot_distractor),
           "musique": ("musique_dev.jsonl", load_musique)}


def _read_records(path):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _slug(s):
    return s.replace("/", "_").replace(":", "_")


def _sample_records(records, n, seed=0):
    """Sample raw records BEFORE building the corpus, so the distractor pool
    is scoped to the sampled questions (not the entire dev set)."""
    if not n or n >= len(records):
        return records
    import random as _r
    rng = _r.Random(seed)
    idx = sorted(rng.sample(range(len(records)), n))
    return [records[i] for i in idx]


def _f1_per_q(report):
    return [r.f1 for r in report.records if r.answerable and r.f1 is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["offline", "real"], default="real")
    ap.add_argument("--datasets", default="hotpotqa")
    ap.add_argument("--models", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--configs", default="naive,reranker_baseline,full")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--unanswerable-frac", type=float, default=0.25)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--llm-planner", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-corrections", type=int, default=2)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results/matrix")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    prefer_fallback = args.backend == "offline"
    os.makedirs(args.out_dir, exist_ok=True)

    for model in models:
        chat = None
        if args.backend == "real":
            from srag import TransformersChat
            chat = TransformersChat(model, max_new_tokens=args.max_new_tokens)
        entailment = EntailmentModel(prefer_fallback=prefer_fallback)
        mslug = _slug(model)

        for dname in datasets:
            fname, loader = LOADERS[dname]
            path = os.path.join(args.data_dir, fname)
            if not os.path.exists(path):
                print(f"[skip] {dname}: missing {path}", flush=True)
                continue
            records = _sample_records(_read_records(path), args.n, seed=0)
            documents, dataset = loader(records)
            # Build a held-out unanswerable subset (these mirrors are
            # answerable-only) so abstention is measured on every dataset.
            if args.unanswerable_frac > 0:
                documents, dataset = make_unanswerable_holdout(
                    documents, dataset, frac=args.unanswerable_frac, seed=0)
            n_dev = int(round(len(dataset) * args.dev_frac))
            dev, test = dataset[:n_dev], dataset[n_dev:]

            thresholds = None
            if dev:
                base = build_config("full", documents, pre_chunked=True,
                                    prefer_fallback=prefer_fallback, chat=chat,
                                    llm_planner=args.llm_planner, entailment=entailment,
                                    max_corrections=args.max_corrections)
                cal = calibrate_thresholds(collect_points(base, dev))
                thresholds = (cal.tau_answer, cal.tau_abstain)
            print(f"[{dname} | {model}] {len(test)} test "
                  f"({sum(1 for d in test if d['answerable'])} ans), "
                  f"{len(documents)} chunks, tau={thresholds}", flush=True)

            cell_reports = {}
            for cfg in configs:
                cell = f"{dname}__{mslug}__{cfg}"
                ckpt = os.path.join(args.out_dir, cell + ".json")
                if os.path.exists(ckpt):
                    print(f"  [cached] {cell}", flush=True)
                    cell_reports[cfg] = json.load(open(ckpt))
                    continue
                t0 = time.time()
                th = thresholds if cfg.startswith("full") else None
                runner = build_config(cfg, documents, pre_chunked=True,
                                      prefer_fallback=prefer_fallback, chat=chat,
                                      thresholds=th, llm_planner=args.llm_planner,
                                      entailment=entailment,
                                      max_corrections=args.max_corrections)
                rep = evaluate_config(runner, test, entailment=entailment,
                                      seeds=1, n_boot=2000)
                payload = {"cell": cell, "dataset": dname, "model": model,
                           "config": cfg, "metrics": rep.to_dict(),
                           "f1_per_q": _f1_per_q(rep),
                           "n_chunks": len(documents), "n_test": len(test),
                           "seconds": round(time.time() - t0, 1)}
                json.dump(payload, open(ckpt, "w"), indent=2)
                cell_reports[cfg] = payload
                d = rep.to_dict()
                print(f"  [done] {cell}  EM={d['EM'][0]:.3f} F1={d['F1'][0]:.3f} "
                      f"abstP/R={d['abstention_P']:.2f}/{d['abstention_R']:.2f} "
                      f"recall@k={d['recall@k'][0]:.3f} ops={d['ops']:.1f} "
                      f"({payload['seconds']:.0f}s)", flush=True)
                _write_summary(args.out_dir)

    _write_summary(args.out_dir, final=True)
    print("\nMatrix complete ->", args.out_dir, flush=True)


def _write_summary(out_dir, final=False):
    cells = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json") and "__" in fn:
            cells.append(json.load(open(os.path.join(out_dir, fn))))
    # group by (dataset, model) for significance full-vs-baseline
    sig = {}
    by_dm = {}
    for c in cells:
        by_dm.setdefault((c["dataset"], c["model"]), {})[c["config"]] = c
    for (ds, model), confs in by_dm.items():
        if "full" in confs and "reranker_baseline" in confs:
            sf, bf = confs["full"]["f1_per_q"], confs["reranker_baseline"]["f1_per_q"]
            if len(sf) == len(bf) and sf:
                p = paired_bootstrap_pvalue(sf, bf, seed=0)
                delta = (sum(sf) / len(sf)) - (sum(bf) / len(bf))
                sig[f"{ds} | {model}"] = {"delta_f1": round(delta, 4),
                                          "p_value": round(p, 4)}
    rows = [{"cell": c["cell"], "dataset": c["dataset"], "model": c["model"],
             "config": c["config"], **c["metrics"]} for c in cells]
    json.dump({"rows": rows, "significance": sig}, open(
        os.path.join(out_dir, "summary.json"), "w"), indent=2)
    if final:
        _write_markdown_summary(out_dir, rows, sig)


def _write_markdown_summary(out_dir, rows, sig):
    lines = ["# Benchmark matrix — summary\n",
             "Real models on real datasets. F1/EM are answer metrics; abst P/R is",
             "abstention precision/recall on the unanswerable subset; ops is mean",
             "operations per query (cost). CIs are 95% bootstraps.\n",
             "| dataset | model | config | EM | F1 | faithful | recall@k | abst P/R | ops |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['model'].split('/')[-1]} | {r['config']} "
            f"| {r['EM'][0]:.3f} | {r['F1'][0]:.3f} | {r['faithfulness'][0]:.3f} "
            f"| {r['recall@k'][0]:.3f} | {r['abstention_P']:.2f}/{r['abstention_R']:.2f} "
            f"| {r['ops']:.1f} |")
    lines.append("\n## Full vs reranked baseline (paired bootstrap)\n")
    lines.append("| dataset | model | Δ F1 | p (one-sided) |")
    lines.append("|---|---|---|---|")
    for k, v in sig.items():
        ds, model = k.split(" | ")
        lines.append(f"| {ds} | {model.split('/')[-1]} | {v['delta_f1']:+.3f} | {v['p_value']:.3f} |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
