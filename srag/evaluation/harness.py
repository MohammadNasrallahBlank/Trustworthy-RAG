"""Run a config over a dataset and aggregate metrics (doc section 7).

Reports correctness, faithfulness, retrieval, abstention, and cost SEPARATELY,
each with a seeded bootstrap CI. Latency is honest wall-clock around the run
(there are no artificial sleeps anywhere in the measured path). Per-question
records are retained so the report can run a paired significance test of the
full system vs. the reranked baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..entailment import EntailmentModel
from ..state import RAGState
from .metrics import (
    bootstrap_ci,
    exact_match,
    reciprocal_rank,
    retrieval_recall_at_k,
    token_f1,
)

# Trace events that count as work for the cost axis.
_COST_EVENTS = {"retrieve", "rerank", "generate", "verify", "correct"}


@dataclass
class QuestionRecord:
    question: str
    type: str
    answerable: bool
    prediction: str
    abstained: bool
    em: Optional[float]              # None for unanswerable items
    f1: Optional[float]
    faithfulness: Optional[float]    # None when no cited claims were emitted
    recall: Optional[float]
    mrr: Optional[float]
    operations: int
    latency_ms: float
    corrections: int


@dataclass
class ConfigReport:
    name: str
    n: int
    # (mean, lo, hi) bootstrap CIs
    em: tuple
    f1: tuple
    faithfulness: tuple
    recall_at_k: tuple
    mrr: tuple
    abstention_precision: float
    abstention_recall: float
    answer_rate_answerable: float
    mean_operations: float
    mean_latency_ms: float
    mean_corrections: float
    records: list = field(default_factory=list)

    def to_dict(self) -> dict:
        f = lambda t: [round(x, 4) for x in t]
        return {
            "config": self.name,
            "n": self.n,
            "EM": f(self.em),
            "F1": f(self.f1),
            "faithfulness": f(self.faithfulness),
            "recall@k": f(self.recall_at_k),
            "MRR": f(self.mrr),
            "abstention_P": round(self.abstention_precision, 4),
            "abstention_R": round(self.abstention_recall, 4),
            "answer_rate": round(self.answer_rate_answerable, 4),
            "ops": round(self.mean_operations, 2),
            "latency_ms": round(self.mean_latency_ms, 3),
            "corrections": round(self.mean_corrections, 3),
        }


def _faithfulness(state: RAGState, entailment: EntailmentModel,
                  threshold: float = 0.5) -> Optional[float]:
    claims = [c for c in state.claims if c.citations]
    if not claims:
        return None
    by_id = {c.id: c for c in state.evidence}
    entailed = 0
    for cl in claims:
        best = max(
            (entailment.entailment(by_id[cid].text, cl.text)
             for cid in cl.citations if cid in by_id),
            default=0.0,
        )
        if best >= threshold:
            entailed += 1
    return entailed / len(claims)


def _evaluate_once(runner, dataset, entailment, k) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    for item in dataset:
        t0 = time.perf_counter()
        try:
            state = runner.run(item["question"])
        except Exception as exc:  # noqa: BLE001 - never let one item kill a long run
            from ..state import RAGState
            state = RAGState(question=item["question"])
            state.abstained = True
            state.answer_status = "abstained"
            state.final_answer = ""
            state.log("error", error=str(exc)[:200])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answerable = bool(item["answerable"])
        golds = item.get("answers", [])
        gold_ids = item.get("supporting_chunk_ids", [])
        pred = "" if state.abstained else state.final_answer
        retrieved_ids = [c.id for c in state.evidence]

        em = f1 = recall = mrr = None
        if answerable:
            em = 0.0 if state.abstained else exact_match(pred, golds)
            f1 = 0.0 if state.abstained else token_f1(pred, golds)
            if gold_ids:
                recall = retrieval_recall_at_k(retrieved_ids, gold_ids, k)
                mrr = reciprocal_rank(retrieved_ids, gold_ids)

        ops = sum(1 for e in state.trace if e["event"] in _COST_EVENTS)
        records.append(QuestionRecord(
            question=item["question"], type=item.get("type", "unknown"),
            answerable=answerable, prediction=pred, abstained=state.abstained,
            em=em, f1=f1, faithfulness=_faithfulness(state, entailment),
            recall=recall, mrr=mrr, operations=ops, latency_ms=latency_ms,
            corrections=state.correction_count,
        ))
    return records


def _avg_records(per_seed: list[list[QuestionRecord]]) -> list[QuestionRecord]:
    """Average numeric metrics across seeds, aligned by question index."""
    if len(per_seed) == 1:
        return per_seed[0]
    out: list[QuestionRecord] = []
    n = len(per_seed[0])
    for i in range(n):
        rs = [seed[i] for seed in per_seed]
        base = rs[0]

        def avg(attr):
            vals = [getattr(r, attr) for r in rs if getattr(r, attr) is not None]
            return sum(vals) / len(vals) if vals else None

        out.append(QuestionRecord(
            question=base.question, type=base.type, answerable=base.answerable,
            prediction=base.prediction, abstained=base.abstained,
            em=avg("em"), f1=avg("f1"), faithfulness=avg("faithfulness"),
            recall=avg("recall"), mrr=avg("mrr"),
            operations=avg("operations") or 0, latency_ms=avg("latency_ms") or 0.0,
            corrections=avg("corrections") or 0,
        ))
    return out


def evaluate_config(runner, dataset: Sequence[dict], *,
                    entailment: Optional[EntailmentModel] = None,
                    seeds: int = 1, k: Optional[int] = None,
                    n_boot: int = 2000) -> ConfigReport:
    entailment = entailment or EntailmentModel(prefer_fallback=True)
    per_seed = [_evaluate_once(runner, dataset, entailment, k) for _ in range(max(1, seeds))]
    records = _avg_records(per_seed)

    answerable = [r for r in records if r.answerable]
    em_vals = [r.em for r in answerable if r.em is not None]
    f1_vals = [r.f1 for r in answerable if r.f1 is not None]
    faith_vals = [r.faithfulness for r in records if r.faithfulness is not None]
    recall_vals = [r.recall for r in answerable if r.recall is not None]
    mrr_vals = [r.mrr for r in answerable if r.mrr is not None]

    # Abstention: positive class = unanswerable; predicted positive = abstained.
    tp = sum(1 for r in records if r.abstained and not r.answerable)
    fp = sum(1 for r in records if r.abstained and r.answerable)
    fn = sum(1 for r in records if not r.abstained and not r.answerable)
    abst_p = tp / (tp + fp) if (tp + fp) else 1.0
    abst_r = tp / (tp + fn) if (tp + fn) else 1.0
    answered = sum(1 for r in answerable if not r.abstained)
    answer_rate = answered / len(answerable) if answerable else 0.0

    return ConfigReport(
        name=runner.name, n=len(records),
        em=bootstrap_ci(em_vals, n_boot=n_boot),
        f1=bootstrap_ci(f1_vals, n_boot=n_boot),
        faithfulness=bootstrap_ci(faith_vals, n_boot=n_boot),
        recall_at_k=bootstrap_ci(recall_vals, n_boot=n_boot),
        mrr=bootstrap_ci(mrr_vals, n_boot=n_boot),
        abstention_precision=abst_p, abstention_recall=abst_r,
        answer_rate_answerable=answer_rate,
        mean_operations=sum(r.operations for r in records) / len(records),
        mean_latency_ms=sum(r.latency_ms for r in records) / len(records),
        mean_corrections=sum(r.corrections for r in records) / len(records),
        records=records,
    )
