"""Reproducibility artifact writer (plan v2 section 3.8).

Every experiment writes a self-describing run directory so a result can be
audited and reproduced:

  manifest.json   -- meta (dataset/model/prompt/retriever/reranker/thresholds/
                     seed) + summary metrics (with CIs)
  examples.jsonl  -- one row per question: {question, gold, evidence ids+text,
                     answer, abstained, category, confidence}
  report.md       -- a human-readable summary

`RunArtifact.save(dir, meta, per_example, summary)` writes it; `RunArtifact.load`
reads it back. `meta` must carry every key in REQUIRED_META_KEYS so a run can
never be silently under-described.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

REQUIRED_META_KEYS = (
    "dataset_version",
    "model_name",
    "prompt_template",
    "retriever_config",
    "reranker_config",
    "verifier_thresholds",
    "seed",
)


@dataclass
class RunArtifact:
    meta: dict
    per_example: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @staticmethod
    def save(dirpath: str, *, meta: dict, per_example: list, summary: dict) -> "RunArtifact":
        missing = [k for k in REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise ValueError(f"artifact meta missing required keys: {missing}")
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": summary}, f, indent=2, default=str)

        with open(os.path.join(dirpath, "examples.jsonl"), "w", encoding="utf-8") as f:
            for ex in per_example:
                f.write(json.dumps(ex, default=str) + "\n")

        with open(os.path.join(dirpath, "report.md"), "w", encoding="utf-8") as f:
            f.write(_render_report(meta, per_example, summary))

        return RunArtifact(meta=meta, per_example=list(per_example), summary=summary)

    @classmethod
    def load(cls, dirpath: str) -> "RunArtifact":
        with open(os.path.join(dirpath, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        per_example = []
        ex_path = os.path.join(dirpath, "examples.jsonl")
        if os.path.exists(ex_path):
            with open(ex_path, encoding="utf-8") as f:
                per_example = [json.loads(line) for line in f if line.strip()]
        return cls(meta=manifest.get("meta", {}),
                   per_example=per_example,
                   summary=manifest.get("summary", {}))


def _render_report(meta: dict, per_example: list, summary: dict) -> str:
    lines = ["# Run report", ""]
    lines.append("## Configuration")
    for k in REQUIRED_META_KEYS:
        lines.append(f"- **{k}**: `{meta.get(k)}`")
    lines.append("")
    lines.append("## Summary metrics")
    for k, v in summary.items():
        if k == "cis":
            continue
        lines.append(f"- **{k}**: {v}")
    cis = summary.get("cis") or {}
    if cis:
        lines.append("")
        lines.append("### 95% bootstrap CIs")
        for k, ci in cis.items():
            try:
                mean, lo, hi = ci
                lines.append(f"- **{k}**: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
            except Exception:
                lines.append(f"- **{k}**: {ci}")
    lines.append("")
    lines.append(f"## Examples ({len(per_example)})")
    lines.append("First few categorized responses:")
    lines.append("")
    lines.append("| question | category | abstained | confidence |")
    lines.append("|---|---|---|---|")
    for ex in per_example[:10]:
        q = str(ex.get("question", ""))[:60].replace("|", "/")
        lines.append(f"| {q} | {ex.get('category','')} | "
                     f"{ex.get('abstained','')} | {ex.get('confidence','')} |")
    lines.append("")
    return "\n".join(lines)
