"""Trustworthy-RAG experiment runner (plan v2 Task 6 / 6b).

Compares the five systems (plan section 3.1) on the trust axis -- Unsupported
Confident Answers vs answerable coverage -- with calibrated abstention, leakage-
checked unanswerables, frozen dev/test discipline, operating curves, and a
reproducibility artifact per system.

Offline smoke (no GPU, no model weights):

    python examples/run_trust.py --dataset docs --backend offline --n 16

Real run (M2, on the GPU box) swaps in a chat LLM via --backend real.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from srag.calibration import calibrate_thresholds, collect_points  # noqa: E402
from srag.evaluation.configs import build_trust_config, TRUST_CONFIGS  # noqa: E402
from srag.evaluation.curves import (  # noqa: E402
    coverage_uca_curve, pick_operating_point, uca_at_coverage, plot_curves,
)
from srag.evaluation.datasets import (  # noqa: E402
    load_docs_eval, load_hotpot_distractor, make_unanswerable_holdout,
    sample_dataset,
)
from srag.evaluation.leakage import verify_unanswerable  # noqa: E402
from srag.evaluation.trust import evaluate_trust  # noqa: E402
from srag.evaluation.artifact import RunArtifact  # noqa: E402

# Systems that produce a coverage/UCA *curve* (thresholded on confidence).
THRESHOLDED = {"retrieval_score_abstain", "guarded", "guarded+correct"}


# --------------------------------------------------------------------------- #
# Data preparation (with leakage-checked unanswerables)
# --------------------------------------------------------------------------- #
def prepare_data(dataset: str, *, n=None, unanswerable_frac=0.4, seed=0):
    if dataset == "docs":
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "data", "docs_eval.jsonl")
        documents, data = load_docs_eval(path)
        # Unanswerables are authored out-of-corpus; still leakage-check them.
        unans = [d for d in data if not d["answerable"]]
        kept, dropped = verify_unanswerable(documents, unans)
        keep_qs = {d["question"] for d in kept}
        data = [d for d in data if d["answerable"] or d["question"] in keep_qs]
        return documents, data, {"dropped_unanswerable": len(dropped)}

    if dataset == "hotpotqa":
        here = os.path.dirname(os.path.abspath(__file__))
        raw = os.path.join(here, os.pardir, "data", "raw", "hotpotqa_dev.json")
        sample = os.path.join(here, os.pardir, "data", "hotpot_sample.json")
        src = raw if os.path.exists(raw) else sample
        records = json.load(open(src, encoding="utf-8"))
        # Sample RAW records BEFORE building the corpus, so the distractor pool
        # is the sampled questions' contexts only (avoids materializing the full
        # dev-set corpus, which is ~70k chunks / tens of GB of TF-IDF).
        if n and n < len(records):
            import random as _r
            rng = _r.Random(seed)
            records = [records[i] for i in sorted(rng.sample(range(len(records)), n))]
        documents, data = load_hotpot_distractor(records)
        # Hold out gold context to synthesize unanswerables, recording the gold
        # answer so the leakage check can verify it is gone.
        held_docs, held_data = make_unanswerable_holdout(
            documents, data, frac=unanswerable_frac, seed=seed)
        # carry original answers onto the held-out items for the leak check
        orig = {d["question"]: d.get("answers", []) for d in data}
        for d in held_data:
            if not d["answerable"]:
                d["held_out_answers"] = orig.get(d["question"], [])
        unans = [d for d in held_data if not d["answerable"]]
        kept, dropped = verify_unanswerable(held_docs, unans)
        keep_qs = {d["question"] for d in kept}
        held_data = [d for d in held_data
                     if d["answerable"] or d["question"] in keep_qs]
        return held_docs, held_data, {"dropped_unanswerable": len(dropped)}

    raise ValueError(f"unknown dataset {dataset!r}")


def split_dev_test(data, *, dev_frac=0.4, seed=0):
    """Stratified dev/test split (frozen-config discipline, section 3.9)."""
    import random
    rng = random.Random(seed)
    ans = [d for d in data if d["answerable"]]
    una = [d for d in data if not d["answerable"]]
    rng.shuffle(ans)
    rng.shuffle(una)
    n_dev_a = max(1, int(round(len(ans) * dev_frac))) if ans else 0
    n_dev_u = max(1, int(round(len(una) * dev_frac))) if una else 0
    dev = ans[:n_dev_a] + una[:n_dev_u]
    test = ans[n_dev_a:] + una[n_dev_u:]
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def _paired_uca_significance(reports, *, seed=0):
    """Paired test that `guarded`/`guarded+correct` answer fewer UNANSWERABLE
    questions (UCA) than the weaker systems, on the shared question set.

    Returns {"<system>_vs_<baseline>": {delta_uca, mcnemar_p, bootstrap_p, n}}.
    UCA indicator per unanswerable question = 1 if the system answered it.
    """
    from srag.evaluation.metrics import paired_bootstrap_pvalue

    def uca_map(name):
        return {r["question"]: (0.0 if r["abstained"] else 1.0)
                for r in reports[name].records if not r["answerable"]}

    out = {}
    systems = ["guarded", "guarded+correct"]
    baselines = ["plain_rag_always_answer", "prompted_abstain_rag", "retrieval_score_abstain"]
    for sysname in systems:
        if sysname not in reports:
            continue
        sm = uca_map(sysname)
        for base in baselines:
            if base not in reports:
                continue
            bm = uca_map(base)
            qs = sorted(set(sm) & set(bm))
            if not qs:
                continue
            sys_u = [sm[q] for q in qs]
            base_u = [bm[q] for q in qs]
            # McNemar on discordant pairs (base answered & sys abstained vs vice versa).
            b = sum(1 for q in qs if bm[q] == 1 and sm[q] == 0)
            c = sum(1 for q in qs if bm[q] == 0 and sm[q] == 1)
            mcnemar_p = _mcnemar_p(b, c)
            # One-sided bootstrap that base_uca - sys_uca > 0 (sys has fewer UCAs).
            boot_p = paired_bootstrap_pvalue(base_u, sys_u, seed=seed)
            out[f"{sysname}_vs_{base}"] = {
                "delta_uca": (sum(sys_u) - sum(base_u)) / len(qs),
                "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar_p,
                "bootstrap_p": boot_p, "n": len(qs),
            }
    return out


def _mcnemar_p(b, c):
    """Exact two-sided McNemar p-value via the binomial on discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_trust(dataset="docs", *, backend="offline", n=None, unanswerable_frac=0.4,
              dev_frac=0.4, seed=0, out_dir=None, min_coverage=0.8, verbose=True,
              model="Qwen/Qwen2.5-3B-Instruct"):
    prefer_fallback = backend != "real"
    chat = None
    if backend == "real":
        from srag.llm import TransformersChat  # noqa: WPS433
        chat = TransformersChat(model)  # GPU path (M2)

    documents, data, prep = prepare_data(
        dataset, n=n, unanswerable_frac=unanswerable_frac, seed=seed)
    dev, test = split_dev_test(data, dev_frac=dev_frac, seed=seed)

    # ---- calibrate guarded thresholds on DEV only (report on TEST) -------- #
    cal_runner = build_trust_config("guarded", documents,
                                    prefer_fallback=prefer_fallback, chat=chat)
    cal = calibrate_thresholds(collect_points(cal_runner, dev))
    thresholds = (cal.tau_answer, cal.tau_abstain)
    if verbose:
        print(f"[{dataset}] prepared: {len(documents)} docs, "
              f"{len(data)} qs (dev {len(dev)} / test {len(test)}); "
              f"dropped unanswerable: {prep['dropped_unanswerable']}")
        print(f"[{dataset}] calibrated thresholds: {cal.to_dict()}")

    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "trust", dataset)
    os.makedirs(out_dir, exist_ok=True)

    reports = {}
    curves = {}
    summary_rows = []
    for name in TRUST_CONFIGS:
        runner = build_trust_config(name, documents, prefer_fallback=prefer_fallback,
                                    thresholds=thresholds, chat=chat)
        report = evaluate_trust(runner, test, n_boot=500, seed=seed, system=name)
        reports[name] = report

        # Curve (thresholded) or single operating point (fixed).
        recs = [{"confidence": r["confidence"], "answerable": r["answerable"]}
                for r in report.records]
        if name in THRESHOLDED:
            curve = coverage_uca_curve(recs)
            curves[name] = curve
            op = pick_operating_point(curve, min_coverage=min_coverage)
            uca_matched = uca_at_coverage(curve, min_coverage)
        else:
            curves[name] = {"coverage": report.metrics["coverage"],
                            "uca": report.metrics["uca_rate"]}
            op = None
            uca_matched = report.metrics["uca_rate"]

        # Reproducibility artifact for this system.
        meta = {
            "dataset_version": f"{dataset}-v1",
            "model_name": "offline-extractive" if prefer_fallback else model,
            "prompt_template": "abstain-v1" if name == "prompted_abstain_rag" else "grounded-v1",
            "retriever_config": {"fusion": "rrf", "reranker": True},
            "reranker_config": {"backend": "offline" if prefer_fallback else "real"},
            "verifier_thresholds": {"tau_answer": thresholds[0], "tau_abstain": thresholds[1]},
            "seed": seed,
        }
        per_example = [{
            "question": r["question"], "gold": r["answers"],
            "evidence": r.get("evidence", []), "answer": r["prediction"],
            "abstained": r["abstained"], "category": r["category"],
            "confidence": r["confidence"], "diagnosis": r.get("diagnosis", ""),
            "corrections": r.get("corrections", 0),
        } for r in report.records]
        sys_summary = dict(report.metrics)
        sys_summary["cis"] = {k: list(v) for k, v in report.cis.items()}
        sys_summary["uca_at_matched_coverage"] = uca_matched
        if op:
            sys_summary["operating_point"] = op
        RunArtifact.save(os.path.join(out_dir, name.replace("+", "_")),
                         meta=meta, per_example=per_example, summary=sys_summary)

        summary_rows.append({
            "system": name,
            "coverage": round(report.metrics["coverage"], 3),
            "uca_rate": round(report.metrics["uca_rate"], 3),
            "uca_at_0.8cov": round(uca_matched, 3),
            "selective_risk": round(report.metrics["selective_risk"], 3),
            "trustworthy_rate": round(report.metrics["trustworthy_rate"], 3),
            "auroc": round(report.metrics["auroc"], 3),
            "ece": round(report.metrics["ece"], 3),
        })

    # Paired significance: does a system produce significantly fewer UCAs than the
    # baselines, on the SHARED unanswerable set? (plan v2 section 2, criterion 1.)
    significance = _paired_uca_significance(reports, seed=seed)

    # Combined summary + curve figure.
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "backend": backend,
                   "thresholds": {"tau_answer": thresholds[0], "tau_abstain": thresholds[1]},
                   "rows": summary_rows, "significance": significance}, f, indent=2)
    png = plot_curves(curves, os.path.join(out_dir, "coverage_uca.png"),
                      title=f"Coverage vs UCA - {dataset}")

    # Failure analysis across every system's per-question records (Task 8).
    try:
        from failure_analysis import analyze_failures
        all_records = [r for rep in reports.values() for r in rep.records]
        fa = analyze_failures(all_records, out_md=os.path.join(out_dir, "failures.md"))
    except Exception as exc:  # pragma: no cover - never block the run on analysis
        fa = {"error": str(exc)}

    if verbose:
        print(f"\n[{dataset}] TEST results (lower UCA / selective_risk is better):")
        hdr = f"{'system':<24}{'cov':>6}{'uca':>7}{'uca@.8':>8}{'selrisk':>9}{'trust':>7}{'auroc':>7}"
        print(hdr)
        for row in summary_rows:
            print(f"{row['system']:<24}{row['coverage']:>6}{row['uca_rate']:>7}"
                  f"{row['uca_at_0.8cov']:>8}{row['selective_risk']:>9}"
                  f"{row['trustworthy_rate']:>7}{row['auroc']:>7}")
        if significance:
            print("\nPaired UCA significance (lower UCA vs baseline, McNemar + bootstrap):")
            for k, v in significance.items():
                print(f"  {k:<34} dUCA={v['delta_uca']:+.3f}  "
                      f"mcnemar_p={v['mcnemar_p']:.4g}  boot_p={v['bootstrap_p']:.4g}")
        print(f"\nArtifacts written to {os.path.abspath(out_dir)}"
              + (f" (curve: {os.path.basename(png)})" if png else " (no matplotlib; curve skipped)"))

    return {"out_dir": out_dir, "summary": summary_rows, "reports": reports}


def main():
    ap = argparse.ArgumentParser(description="Trustworthy-RAG experiment runner")
    ap.add_argument("--dataset", choices=["docs", "hotpotqa"], default="docs")
    ap.add_argument("--backend", choices=["offline", "real"], default="offline")
    ap.add_argument("--n", type=int, default=None, help="sample N questions (hotpotqa)")
    ap.add_argument("--unanswerable-frac", type=float, default=0.4)
    ap.add_argument("--dev-frac", type=float, default=0.4)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_trust(args.dataset, backend=args.backend, n=args.n,
              unanswerable_frac=args.unanswerable_frac, dev_frac=args.dev_frac,
              seed=args.seed, out_dir=args.out, min_coverage=args.min_coverage,
              model=args.model)


if __name__ == "__main__":
    main()
