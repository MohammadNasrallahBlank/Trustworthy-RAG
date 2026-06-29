"""Verifier — the heart of the system (doc section 4.6).

Runs a cascade of checks, cheapest first, and produces both a scalar confidence
and a structured *diagnosis* -- never a bare "sufficient/insufficient". Each
check is a standalone scorer (doc roadmap step 2), so they can be unit-tested
and validated in isolation before being wired into the controller (stage 4).

Checks
  1. Retrieval-quality  (cheap)   -- reranker score level + margin (CRAG-style).
  2. Claim attribution  (medium)  -- per-claim NLI against the cited chunk.
  3. Self-consistency   (medium)  -- agreement across sampled answers (optional).
  4. Sub-question cover (cheap)   -- every planned hop has a supporting passage.
  5. Conflict detection (medium)  -- two high-score chunks contradict each other.

Thresholds here are sensible defaults; calibration on a dev set is stage 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .entailment import EntailmentModel
from .state import Chunk, Claim, RAGState, SubQuery

# Diagnosis labels (match the controller's routing table, doc section 4.7).
PASS = "pass"
RETRIEVAL_FAULT = "retrieval_fault"
GENERATION_FAULT = "generation_fault"
PLANNING_FAULT = "planning_fault"
CONFLICT = "conflict"


@dataclass
class CheckResult:
    name: str
    score: float                       # normalized [0,1], higher = healthier
    detail: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Structured verifier output (doc section 4.6 schema)."""

    confidence: float
    diagnosis: str
    failing_hops: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    evidence_seen: bool = True
    conflict: bool = False
    suggested_queries: list[str] = field(default_factory=list)
    checks: dict[str, CheckResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 4),
            "diagnosis": self.diagnosis,
            "failing_hops": self.failing_hops,
            "unsupported_claims": self.unsupported_claims,
            "evidence_seen": self.evidence_seen,
            "conflict": self.conflict,
            "suggested_queries": self.suggested_queries,
        }


# A sampler used by the optional self-consistency check.
Sampler = Callable[[str], str]


class Verifier:
    def __init__(
        self,
        entailment: Optional[EntailmentModel] = None,
        *,
        rerank_floor: float = 0.15,
        margin_floor: float = 0.05,
        entail_threshold: float = 0.5,
        weights: Optional[dict] = None,
    ) -> None:
        self.entailment = entailment or EntailmentModel()
        self.rerank_floor = rerank_floor
        self.margin_floor = margin_floor
        self.entail_threshold = entail_threshold
        # Coverage is applied as a GATE (multiplier) below, not a blend term,
        # so these weights cover only the health signals and sum to 1.0.
        self.weights = weights or {
            "retrieval": 0.55,
            "attribution": 0.30,
            "consistency": 0.15,
        }

    # ================================================================== #
    # Check 1 -- retrieval quality (cheap, pre/post-generation)
    # ================================================================== #
    def check_retrieval_quality(self, evidence: Sequence[Chunk]) -> CheckResult:
        if not evidence:
            return CheckResult("retrieval", 0.0, {"degree": "incorrect", "top": 0.0, "margin": 0.0})
        scores = sorted((c.rerank_score for c in evidence), reverse=True)
        top = scores[0]
        margin = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]
        if top < self.rerank_floor:
            degree = "incorrect"
        elif margin < self.margin_floor:
            degree = "ambiguous"
        else:
            degree = "correct"
        score = max(0.0, min(1.0, top))
        return CheckResult("retrieval", score, {"degree": degree, "top": float(top), "margin": float(margin)})

    # ================================================================== #
    # Check 2 -- claim-level entailment / attribution (medium)
    # ================================================================== #
    def check_attribution(
        self, claims: Sequence[Claim], evidence: Sequence[Chunk]
    ) -> CheckResult:
        by_id = {c.id: c for c in evidence}
        per_claim: list[dict] = []
        unsupported: list[str] = []
        entailed = 0
        for claim in claims:
            best = 0.0
            best_src = None
            premises = (
                [(cid, by_id[cid].text) for cid in claim.citations if cid in by_id]
                or [(c.id, c.text) for c in evidence]
            )
            for cid, premise in premises:
                p = self.entailment.entailment(premise, claim.text)
                if p > best:
                    best, best_src = p, cid
            supported = best >= self.entail_threshold and bool(claim.citations)
            per_claim.append(
                {"claim": claim.text, "entailment": round(best, 3), "best_src": best_src,
                 "cited": bool(claim.citations), "supported": supported}
            )
            if supported:
                entailed += 1
            else:
                unsupported.append(claim.text)
        rate = entailed / len(claims) if claims else 0.0
        return CheckResult(
            "attribution",
            rate,
            {"per_claim": per_claim, "unsupported": unsupported, "n_claims": len(claims)},
        )

    # ================================================================== #
    # Check 3 -- self-consistency (medium, optional)
    # ================================================================== #
    def check_self_consistency(
        self, question: str, sampler: Optional[Sampler], k: int = 5
    ) -> Optional[CheckResult]:
        if sampler is None:
            return None
        answers = [_norm(sampler(question)) for _ in range(k)]
        answers = [a for a in answers if a]
        if not answers:
            return CheckResult("consistency", 0.0, {"answers": []})
        counts: dict[str, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        top = max(counts.values())
        agreement = top / len(answers)
        return CheckResult("consistency", agreement, {"answers": answers, "agreement": agreement})

    # ================================================================== #
    # Check 4 -- sub-question coverage (cheap, multi-hop)
    # ================================================================== #
    def check_coverage(
        self, sub_queries: Sequence[SubQuery], evidence: Sequence[Chunk]
    ) -> CheckResult:
        if not sub_queries:
            return CheckResult("coverage", 1.0, {"failing_hops": [], "per_hop": {}})
        failing: list[str] = []
        per_hop: dict[str, float] = {}
        for sq in sub_queries:
            # An unresolved hop is templated on an earlier hop's answer that is
            # not yet known, so it cannot be covered until the controller fills
            # it. Its best-effort `text` may incidentally lexically match the
            # prerequisite hop's evidence, so we do NOT score it -- it is failing
            # by construction.
            if not sq.resolved:
                per_hop[sq.id] = 0.0
                failing.append(sq.id)
                continue
            best = max((self._supports(sq.text, c.text) for c in evidence), default=0.0)
            per_hop[sq.id] = round(best, 3)
            if best < 0.2:
                failing.append(sq.id)
        covered = len(sub_queries) - len(failing)
        return CheckResult(
            "coverage",
            covered / len(sub_queries),
            {"failing_hops": failing, "per_hop": per_hop},
        )

    # ================================================================== #
    # Check 5 -- conflict detection (medium)
    # ================================================================== #
    def check_conflict(self, evidence: Sequence[Chunk]) -> CheckResult:
        top = [c for c in evidence if c.rerank_score >= self.rerank_floor][:5]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                if self._contradicts(top[i].text, top[j].text):
                    return CheckResult(
                        "conflict", 1.0,
                        {"conflict": True, "between": [top[i].id, top[j].id]},
                    )
        return CheckResult("conflict", 0.0, {"conflict": False, "between": []})

    # ================================================================== #
    # Aggregation -> confidence + diagnosis
    # ================================================================== #
    def verify(
        self,
        state: RAGState,
        *,
        sampler: Optional[Sampler] = None,
        self_consistency_k: int = 5,
    ) -> VerificationResult:
        evidence = state.evidence
        claims = state.claims

        c_ret = self.check_retrieval_quality(evidence)
        c_attr = self.check_attribution(claims, evidence)
        c_cov = self.check_coverage(state.sub_queries, evidence)
        c_con = self.check_conflict(evidence)
        c_cons = self.check_self_consistency(state.question, sampler, self_consistency_k)

        checks = {c.name: c for c in [c_ret, c_attr, c_cov, c_con] if c is not None}
        if c_cons is not None:
            checks["consistency"] = c_cons

        # Blend the health signals, then GATE by coverage: a partially-covered
        # multi-hop answer cannot be high-confidence no matter how well its
        # present claims are attributed -- an uncovered required hop means the
        # answer is still incomplete. Conflict applies a further penalty.
        w = dict(self.weights)
        w.pop("coverage", None)
        if c_cons is None:
            extra = w.pop("consistency", 0.0)
            total = sum(w.values()) or 1.0
            w = {k: v + extra * (v / total) for k, v in w.items()}
        blend = (
            w.get("retrieval", 0) * c_ret.score
            + w.get("attribution", 0) * c_attr.score
            + (w.get("consistency", 0) * c_cons.score if c_cons is not None else 0.0)
        )
        confidence = blend * (0.4 + 0.6 * c_cov.score)
        if c_con.detail["conflict"]:
            confidence *= 0.6

        unsupported = c_attr.detail["unsupported"]
        failing_hops = c_cov.detail["failing_hops"]
        evidence_seen = c_ret.detail["degree"] != "incorrect"

        # Diagnosis priority (doc sections 4.6/4.7).
        if not state.answerable and not claims:
            diagnosis = RETRIEVAL_FAULT
        elif c_con.detail["conflict"]:
            diagnosis = CONFLICT
        elif failing_hops and not evidence_seen:
            diagnosis = RETRIEVAL_FAULT
        elif failing_hops and evidence_seen:
            diagnosis = PLANNING_FAULT
        elif unsupported and evidence_seen:
            diagnosis = GENERATION_FAULT
        elif unsupported and not evidence_seen:
            diagnosis = RETRIEVAL_FAULT
        else:
            diagnosis = PASS

        suggested = self._suggest_queries(state, failing_hops, diagnosis)

        result = VerificationResult(
            confidence=float(confidence),
            diagnosis=diagnosis,
            failing_hops=failing_hops,
            unsupported_claims=unsupported,
            evidence_seen=evidence_seen,
            conflict=c_con.detail["conflict"],
            suggested_queries=suggested,
            checks=checks,
        )
        state.confidence = result.confidence
        state.diagnosis = result.diagnosis
        state.failing_hops = result.failing_hops
        state.unsupported_claims = result.unsupported_claims
        state.conflict = result.conflict
        state.suggested_queries = result.suggested_queries
        state.log("verify", **result.to_dict())
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _suggest_queries(
        self, state: RAGState, failing_hops: list[str], diagnosis: str
    ) -> list[str]:
        if diagnosis not in (RETRIEVAL_FAULT, PLANNING_FAULT):
            return []
        hop_map = {sq.id: sq for sq in state.sub_queries}
        if failing_hops:
            return [hop_map[h].text for h in failing_hops if h in hop_map]
        return [state.question]

    def _supports(self, query: str, passage: str) -> float:
        return self.entailment.entailment(passage, query)

    def _contradicts(self, a: str, b: str) -> bool:
        ta, tb = _content_terms(a), _content_terms(b)
        shared = ta & tb
        if len(shared) < 2:
            return False
        a_neg = _has_neg(a)
        b_neg = _has_neg(b)
        return a_neg != b_neg


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it", "from",
}
_NEG = {"not", "no", "never", "none", "cannot"}


def _content_terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _has_neg(text: str) -> bool:
    toks = set(_TOKEN.findall(text.lower()))
    return bool(_NEG & toks) or "n't" in text.lower()


def _norm(s: str) -> str:
    return " ".join(_TOKEN.findall((s or "").lower()))
