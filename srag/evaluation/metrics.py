"""Evaluation metrics (doc section 7), reported separately.

  * Answer correctness: SQuAD/Hotpot-style EM and token-F1.
  * Retrieval quality:  recall@k and MRR against gold supporting chunks.
  * Faithfulness:       handled at the harness level via claim-level entailment.
  * Statistics:         seeded bootstrap CIs and a paired bootstrap significance
                        test for the key full-vs-baseline comparison.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Callable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation/articles/extra space."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(g) for g in golds) else 0.0


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Max token-level F1 of the prediction against any gold answer."""
    pred_toks = normalize_answer(prediction).split()
    best = 0.0
    for g in golds:
        gold_toks = normalize_answer(g).split()
        if not pred_toks and not gold_toks:
            best = max(best, 1.0)
            continue
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(pred_toks)
        recall = n_same / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def retrieval_recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str],
                          k: int | None = None) -> float:
    """Fraction of gold supporting chunks present in the top-k retrieved set."""
    if not gold_ids:
        return 1.0
    topk = set(retrieved_ids[:k] if k else retrieved_ids)
    hit = sum(1 for g in gold_ids if g in topk)
    return hit / len(gold_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """1/rank of the first gold chunk in the retrieved list (0 if absent)."""
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------- #
# Statistics
# ---------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap CI over `values`."""
    vals = list(values)
    if not vals:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / len(vals)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (mean, lo, hi)


def paired_bootstrap_pvalue(system: Sequence[float], baseline: Sequence[float],
                            *, n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided paired bootstrap p-value for mean(system) > mean(baseline).

    Resamples question indices with replacement and measures how often the
    system's mean does NOT exceed the baseline's. Lower p == stronger evidence
    the full system beats the baseline.
    """
    assert len(system) == len(baseline), "paired test needs aligned per-question scores"
    diffs = [s - b for s, b in zip(system, baseline)]
    n = len(diffs)
    if n == 0:
        return 1.0
    rng = random.Random(seed)
    not_better = 0
    for _ in range(n_boot):
        sample_mean = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        if sample_mean <= 0:
            not_better += 1
    return not_better / n_boot
