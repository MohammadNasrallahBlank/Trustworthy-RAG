"""Failure analysis for a trust run (plan v2 Task 8).

Reads the per-example records from a trust run (the reproducibility artifacts'
`examples.jsonl`, or an in-memory record list) and auto-tags every FAILURE
(uca / wrong / missed) into a small set of actionable classes, so M3 hardening
targets what the data actually shows -- not guesses.

Tag rules (cheap, evidence-based; an optional LLM judge can refine later):
  retrieval_missed_evidence      gold answer is NOT in the retrieved evidence.
  evidence_ignored_by_generator  gold answer IS in evidence, yet answered wrong.
  verifier_too_strict            answerable + abstained, but evidence had the gold.
  verifier_too_lenient           unanswerable answered with high confidence.
  correction_retrieved_irrelevant correction(s) fired but gold still absent.
  answered_from_prior_knowledge  unanswerable answered, gold/answer not in evidence.
  ambiguous_question             everything else (fallback).

Usage:
    python tools/failure_analysis.py results/trust/hotpotqa --out results/trust/hotpotqa/failures.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag.evaluation.metrics import normalize_answer  # noqa: E402

FAILURE_TAGS = [
    "retrieval_missed_evidence",
    "evidence_ignored_by_generator",
    "verifier_too_strict",
    "verifier_too_lenient",
    "correction_retrieved_irrelevant",
    "answered_from_prior_knowledge",
    "ambiguous_question",
]

FAILURE_CATEGORIES = {"uca", "wrong", "missed"}


def _evidence_text(rec: dict) -> str:
    parts = []
    for e in rec.get("evidence", []) or []:
        parts.append(e["text"] if isinstance(e, dict) else getattr(e, "text", ""))
    return normalize_answer(" ".join(parts))


def _gold_in_evidence(rec: dict) -> bool:
    ev = _evidence_text(rec)
    if not ev:
        return False
    for g in rec.get("answers", []) or []:
        gn = normalize_answer(str(g))
        if gn and f" {gn} " in f" {ev} ":
            return True
    return False


def tag_failure(rec: dict, *, lenient_conf: float = 0.9) -> str:
    """Assign one failure tag to a single failing record."""
    cat = rec.get("category")
    gold_present = _gold_in_evidence(rec)

    if cat == "uca":
        # Answered an unanswerable. High confidence -> the verifier should have
        # caught it; otherwise it's a parametric (prior-knowledge) answer.
        if float(rec.get("confidence", 0.0)) >= lenient_conf:
            return "verifier_too_lenient"
        return "answered_from_prior_knowledge"

    if cat == "missed":
        # Abstained on an answerable question.
        if gold_present:
            return "verifier_too_strict"
        return "retrieval_missed_evidence"

    if cat == "wrong":
        if not gold_present:
            # Evidence never contained the answer.
            if int(rec.get("corrections", 0) or 0) > 0:
                return "correction_retrieved_irrelevant"
            return "retrieval_missed_evidence"
        return "evidence_ignored_by_generator"

    return "ambiguous_question"


def analyze_failures(records, *, out_md=None, sample_per_tag: int = 4) -> dict:
    """Tag all failing records, count by tag, and (optionally) write a report."""
    failures = [r for r in records if r.get("category") in FAILURE_CATEGORIES]
    tagged = []
    counts: dict[str, int] = {}
    examples: dict[str, list] = {t: [] for t in FAILURE_TAGS}
    for r in failures:
        tag = tag_failure(r)
        counts[tag] = counts.get(tag, 0) + 1
        tagged.append({**r, "failure_tag": tag})
        if len(examples[tag]) < sample_per_tag:
            examples[tag].append(r)

    summary = {"n_total": len(records), "n_failures": len(failures),
               "counts": counts, "tagged": tagged}

    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(_render(summary, examples))
    return summary


def _render(summary: dict, examples: dict) -> str:
    lines = ["# Failure analysis", ""]
    lines.append(f"- Responses analyzed: **{summary['n_total']}**")
    lines.append(f"- Failures (uca + wrong + missed): **{summary['n_failures']}**")
    lines.append("")
    lines.append("## Failure classes")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for tag in FAILURE_TAGS:
        c = summary["counts"].get(tag, 0)
        if c:
            lines.append(f"| {tag} | {c} |")
    lines.append("")
    for tag in FAILURE_TAGS:
        exs = examples.get(tag, [])
        if not exs:
            continue
        lines.append(f"### {tag} ({summary['counts'].get(tag, 0)})")
        for r in exs:
            q = str(r.get("question", ""))[:100]
            gold = ", ".join(map(str, r.get("answers", []) or [])) or "(none)"
            lines.append(f"- **Q:** {q}")
            lines.append(f"  - predicted: `{r.get('prediction','')}` | gold: `{gold}` "
                         f"| conf: {r.get('confidence','')} | corrections: {r.get('corrections',0)}")
        lines.append("")
    return "\n".join(lines)


def _load_records(run_dir: str) -> list:
    """Collect per-example records from every system artifact in a run dir."""
    records = []
    for name in os.listdir(run_dir):
        ex = os.path.join(run_dir, name, "examples.jsonl")
        if os.path.exists(ex):
            with open(ex, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser(description="Tag failures from a trust run")
    ap.add_argument("run_dir", help="results/trust/<dataset> directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    records = _load_records(args.run_dir)
    out = args.out or os.path.join(args.run_dir, "failures.md")
    summary = analyze_failures(records, out_md=out)
    print(f"Analyzed {summary['n_total']} responses, {summary['n_failures']} failures.")
    for tag, c in sorted(summary["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {tag:<34}{c}")
    print(f"Report: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
