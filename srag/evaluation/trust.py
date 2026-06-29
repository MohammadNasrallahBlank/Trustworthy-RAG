"""Trustworthy-RAG metrics + the trust evaluation (plan v2 sections 3.2-3.6).

The project's value is *fewer Unsupported Confident Answers (UCA) at a given
answerable coverage*, with well-calibrated abstention. This module turns a run
into the metrics that measure exactly that:

  * categorize()         -> correct / wrong / missed / refusal / uca per response
  * trust_metrics()      -> UCA rate, trustworthy rate, coverage, selective risk,
                            attempted-answer accuracy, FRR, FAR (section 3.5)
  * calibration_metrics()-> AUROC / AUPRC / ECE for the verifier-confidence
                            detector of "answer is supported" (section 3.6)
  * evaluate_trust()     -> run a system over a dataset and assemble a TrustReport

All pure-numpy / pure-python so the offline suite needs no model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .metrics import bootstrap_ci, exact_match, token_f1

# Outcome category labels.
CORRECT = "correct"
WRONG = "wrong"
MISSED = "missed"
REFUSAL = "refusal"
UCA = "uca"

# An answer is "correct" if its token-F1 against any gold meets this floor. EM
# is also accepted (EM == 1 implies F1 == 1).
F1_CORRECT_FLOOR = 0.5


def categorize(*, answerable: bool, abstained: bool, correct: bool) -> str:
    """Map one response to an outcome category (plan v2 section 3.3)."""
    if not answerable:
        return REFUSAL if abstained else UCA
    if abstained:
        return MISSED
    return CORRECT if correct else WRONG


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def trust_metrics(records: Sequence[dict]) -> dict:
    """Aggregate per-response records into the headline trust metrics.

    Each record needs `answerable`, `abstained`, `correct` (bool). Optional
    `em` / `f1` floats feed answer-accuracy on attempted answerable questions.
    """
    n = len(records)
    cats = [categorize(answerable=bool(r["answerable"]),
                        abstained=bool(r["abstained"]),
                        correct=bool(r.get("correct", False))) for r in records]
    n_correct = cats.count(CORRECT)
    n_wrong = cats.count(WRONG)
    n_missed = cats.count(MISSED)
    n_refusal = cats.count(REFUSAL)
    n_uca = cats.count(UCA)

    n_answerable = sum(1 for r in records if r["answerable"])
    n_unanswerable = n - n_answerable
    n_attempted = n_correct + n_wrong          # non-abstained answerable

    # EM/F1 reported on attempted answerable only (section 3.5).
    attempted = [r for r, c in zip(records, cats) if c in (CORRECT, WRONG)]
    em = _safe_div(sum(float(r.get("em", 1.0 if c == CORRECT else 0.0))
                       for r, c in zip(records, cats) if c in (CORRECT, WRONG)),
                   len(attempted))
    f1 = _safe_div(sum(float(r.get("f1", 1.0 if c == CORRECT else 0.0))
                       for r, c in zip(records, cats) if c in (CORRECT, WRONG)),
                   len(attempted))

    return {
        "n": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "counts": {CORRECT: n_correct, WRONG: n_wrong, MISSED: n_missed,
                   REFUSAL: n_refusal, UCA: n_uca},
        # Headline.
        "uca_rate": _safe_div(n_uca, n_unanswerable),
        "uca_among_unanswerable": _safe_div(n_uca, n_unanswerable),
        "trustworthy_rate": 1.0 - _safe_div(n_wrong + n_uca, n) if n else 0.0,
        "coverage": _safe_div(n_attempted, n_answerable),
        # Selective risk (section 3.5).
        "attempted_accuracy": _safe_div(n_correct, n_attempted),
        "selective_risk": _safe_div(n_wrong, n_attempted),
        # Calibration-style rates.
        "frr": _safe_div(n_missed, n_answerable),   # false refusal (answerable)
        "far": _safe_div(n_uca, n_unanswerable),    # false accept (unanswerable)
        "answer_em": em,
        "answer_f1": f1,
    }


def frr(records: Sequence[dict]) -> float:
    """False refusal rate: abstained-on-answerable / answerable."""
    return trust_metrics(records)["frr"]


def far(records: Sequence[dict]) -> float:
    """False accept rate: non-abstained-on-unanswerable / unanswerable."""
    return trust_metrics(records)["far"]


# ====================================================================== #
# Calibration metrics (section 3.6) -- pure python.
# ====================================================================== #
def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity, ties handled."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _auprc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the precision-recall curve (trapezoid over recall)."""
    if not any(labels):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(labels)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    prev_precision = 1.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall = recall
        prev_precision = precision
    return area


def _ece(scores: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error over `n_bins` equal-width confidence buckets."""
    n = len(scores)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # last bucket is closed on the right so score == 1.0 lands somewhere.
        idx = [i for i in range(n)
               if (lo <= scores[i] < hi) or (b == n_bins - 1 and scores[i] == hi)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def calibration_metrics(scores: Sequence[float], labels: Sequence[int],
                        n_bins: int = 10) -> dict:
    """AUROC / AUPRC / ECE for a confidence detector of 'answer is supported'."""
    scores = [float(s) for s in scores]
    labels = [int(y) for y in labels]
    return {
        "auroc": _auroc(scores, labels),
        "auprc": _auprc(scores, labels),
        "ece": _ece(scores, labels, n_bins=n_bins),
        "n": len(scores),
        "n_pos": sum(labels),
    }


# ====================================================================== #
# evaluate_trust -- run a system over a dataset and assemble the report.
# ====================================================================== #
@dataclass
class TrustReport:
    system: str
    metrics: dict
    records: list = field(default_factory=list)
    cis: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"system": self.system, "metrics": self.metrics, "cis": self.cis,
                "records": self.records}


def _is_correct(prediction: str, golds: Sequence[str]) -> tuple[bool, float, float]:
    if not golds:
        return (False, 0.0, 0.0)
    em = exact_match(prediction, golds)
    f1 = token_f1(prediction, golds)
    correct = em == 1.0 or f1 >= F1_CORRECT_FLOOR
    return (correct, em, f1)


def evaluate_trust(runner, dataset: Sequence[dict], *, n_boot: int = 2000,
                   seed: int = 0, system: Optional[str] = None) -> TrustReport:
    """Run `runner` over `dataset` and compute the trust + calibration metrics.

    `runner` is a config Runner (name + run(question)->RAGState). Each dataset
    item needs `question`, `answers` (list; [] for unanswerable), `answerable`.
    """
    records: list[dict] = []
    cal_scores: list[float] = []
    cal_labels: list[int] = []

    for item in dataset:
        state = runner.run(item["question"])
        answerable = bool(item.get("answerable", bool(item.get("answers"))))
        abstained = bool(getattr(state, "abstained", False))
        pred = getattr(state, "final_answer", "") or ""
        golds = item.get("answers", []) or []
        correct, em, f1 = _is_correct(pred, golds) if (answerable and not abstained) else (False, 0.0, 0.0)
        conf = getattr(state, "confidence", None)
        conf = float(conf) if conf is not None else 0.0

        cat = categorize(answerable=answerable, abstained=abstained, correct=correct)
        evidence = [{"id": getattr(c, "id", ""),
                     "text": (getattr(c, "text", "") or "")[:400]}
                    for c in (getattr(state, "evidence", []) or [])]
        records.append({
            "question": item["question"],
            "answerable": answerable,
            "abstained": abstained,
            "correct": correct,
            "em": em,
            "f1": f1,
            "prediction": pred,
            "answers": golds,
            "confidence": conf,
            "answer_status": getattr(state, "answer_status", ""),
            "diagnosis": getattr(state, "diagnosis", ""),
            "corrections": int(getattr(state, "correction_count", 0) or 0),
            "evidence": evidence,
            "category": cat,
        })
        # Calibration detector: confidence should rank supported (answerable &
        # correct) answers above everything else.
        cal_scores.append(conf)
        cal_labels.append(1 if (answerable and correct) else 0)

    metrics = trust_metrics(records)
    metrics.update(calibration_metrics(cal_scores, cal_labels))

    # Bootstrap CIs on the headline rates (per-question indicator series).
    cis = {}
    uca_series = [1.0 if r["category"] == UCA else 0.0
                  for r in records if not r["answerable"]]
    cov_series = [0.0 if r["abstained"] else 1.0
                  for r in records if r["answerable"]]
    if uca_series:
        cis["uca_rate"] = bootstrap_ci(uca_series, n_boot=n_boot, seed=seed)
    if cov_series:
        cis["coverage"] = bootstrap_ci(cov_series, n_boot=n_boot, seed=seed)

    return TrustReport(system=system or getattr(runner, "name", "system"),
                       metrics=metrics, records=records, cis=cis)
