"""Threshold calibration for abstention (doc section 4.9 / section 7).

"Calibrate the thresholds on a dev set with a known-unanswerable subset; report
abstention precision/recall, not just answer accuracy."

This fits `tau_answer` and `tau_abstain` for the `Finalizer` from a dev set of
(question, answerable) labels by:
  1. running the controller over the dev set and recording the final confidence,
  2. sweeping candidate thresholds to maximize F1 of the relevant decision
     (abstain-on-unanswerable for tau_abstain; answer-on-answerable for
     tau_answer), with tau_answer constrained >= tau_abstain.

Returns the chosen thresholds plus the precision/recall the doc asks for. The
result is framework-agnostic: pass the confidence/label points directly, or use
`collect_points` to gather them from a live controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    tau_answer: float
    tau_abstain: float
    abstention_precision: float
    abstention_recall: float
    answer_precision: float
    answer_recall: float
    n: int
    points: list[tuple] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tau_answer": round(self.tau_answer, 4),
            "tau_abstain": round(self.tau_abstain, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "abstention_recall": round(self.abstention_recall, 4),
            "answer_precision": round(self.answer_precision, 4),
            "answer_recall": round(self.answer_recall, 4),
            "n": self.n,
        }


def collect_points(controller, dev_set) -> list[tuple]:
    """Run `controller` over the dev set, return [(confidence, answerable)]."""
    points: list[tuple] = []
    for item in dev_set:
        state = controller.run(item["question"])
        conf = state.confidence if state.confidence is not None else 0.0
        points.append((float(conf), bool(item["answerable"])))
    return points


def _candidate_thresholds(confidences: list[float]) -> list[float]:
    xs = sorted(set(confidences))
    cands = [0.0]
    for i in range(len(xs)):
        cands.append(xs[i])
        if i + 1 < len(xs):
            cands.append((xs[i] + xs[i + 1]) / 2.0)
    cands.append(1.0001)
    return sorted(set(cands))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def calibrate_thresholds(points: list[tuple]) -> CalibrationResult:
    """Fit (tau_answer, tau_abstain) from [(confidence, answerable)] points."""
    if not points:
        raise ValueError("calibrate_thresholds needs at least one point.")
    confs = [c for c, _ in points]
    cands = _candidate_thresholds(confs)

    # tau_abstain: predict ABSTAIN when conf < tau; positive = unanswerable.
    best_ab = (-1.0, 0.30, 0.0, 0.0)  # (f1, tau, precision, recall)
    for tau in cands:
        tp = sum(1 for c, ans in points if c < tau and not ans)
        fp = sum(1 for c, ans in points if c < tau and ans)
        fn = sum(1 for c, ans in points if c >= tau and not ans)
        p, r, f1 = _prf(tp, fp, fn)
        # Tie-break: higher F1, then higher recall, then lower tau.
        if (f1, r, -tau) > (best_ab[0], best_ab[3], -best_ab[1]):
            best_ab = (f1, tau, p, r)
    _, tau_abstain, ab_p, ab_r = best_ab

    # tau_answer: predict ANSWER when conf >= tau; positive = answerable.
    # Constrained to be >= tau_abstain so the bands are ordered.
    best_an = (-1.0, max(tau_abstain, 0.55), 0.0, 0.0)
    for tau in cands:
        if tau < tau_abstain:
            continue
        tp = sum(1 for c, ans in points if c >= tau and ans)
        fp = sum(1 for c, ans in points if c >= tau and not ans)
        fn = sum(1 for c, ans in points if c < tau and ans)
        p, r, f1 = _prf(tp, fp, fn)
        if (f1, r, -tau) > (best_an[0], best_an[3], -best_an[1]):
            best_an = (f1, tau, p, r)
    _, tau_answer, an_p, an_r = best_an

    # --- robustness guard -------------------------------------------------
    # If the loop gives answerable items low/overlapping confidence, the
    # optimizer above can pick thresholds that abstain on (almost) everything.
    # Cap the thresholds by the answerable-confidence distribution so the system
    # still ANSWERS a reasonable fraction of answerable questions: tau_abstain
    # must let >=80% of answerable through, and tau_answer must let >=50% through.
    ans_conf = sorted(c for c, a in points if a)
    if ans_conf:
        def _pct(p):
            idx = max(0, min(len(ans_conf) - 1, int(p * len(ans_conf))))
            return ans_conf[idx]
        tau_abstain = min(tau_abstain, _pct(0.20))   # abstain on <=20% of answerable
        tau_answer = min(tau_answer, _pct(0.50))     # answer >=50% of answerable
        tau_answer = max(tau_answer, tau_abstain)    # keep ordering
        # Recompute reported abstention P/R at the guarded tau_abstain.
        tp = sum(1 for c, a in points if c < tau_abstain and not a)
        fp = sum(1 for c, a in points if c < tau_abstain and a)
        fn = sum(1 for c, a in points if c >= tau_abstain and not a)
        ab_p = tp / (tp + fp) if (tp + fp) else 1.0
        ab_r = tp / (tp + fn) if (tp + fn) else 1.0

    return CalibrationResult(
        tau_answer=tau_answer,
        tau_abstain=tau_abstain,
        abstention_precision=ab_p,
        abstention_recall=ab_r,
        answer_precision=an_p,
        answer_recall=an_r,
        n=len(points),
        points=list(points),
    )
