"""Tests for the Stage 2 verifier (Checks 1–5 + aggregation).

All run offline on the lexical entailment fallback. They assert the *diagnostic
behavior* the controller will route on, not exact confidence values:
  - supported claims score higher than unsupported ones (entailment ranking),
  - evidence-present-but-unsupported  -> generation_fault,
  - nothing-on-topic / low retrieval  -> retrieval_fault,
  - an uncovered hop with evidence     -> planning_fault,
  - contradictory high-score chunks    -> conflict,
  - a clean grounded answer            -> pass.
"""

from __future__ import annotations

import pytest

from srag import (
    Chunk,
    Claim,
    EntailmentModel,
    RAGState,
    SubQuery,
    Verifier,
)
from srag.verifier import (
    CONFLICT,
    GENERATION_FAULT,
    PASS,
    PLANNING_FAULT,
    RETRIEVAL_FAULT,
)


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(EntailmentModel(prefer_fallback=True))


def _chunk(cid, text, rerank=0.8):
    c = Chunk(id=cid, text=text)
    c.rerank_score = rerank
    return c


# ---------------------------------------------------------- entailment
def test_entailment_ranks_supported_above_unsupported():
    nli = EntailmentModel(prefer_fallback=True)
    premise = "Akira Kurosawa was a Japanese filmmaker born in Tokyo."
    supported = nli.entailment(premise, "Kurosawa was Japanese")
    unsupported = nli.entailment(premise, "Kurosawa was a French painter from Paris")
    assert supported > unsupported


def test_entailment_penalizes_negation_mismatch():
    nli = EntailmentModel(prefer_fallback=True)
    pos = "The bridge was built in 1673."
    neg_hyp_score = nli.entailment(pos, "The bridge was not built in 1673")
    pos_hyp_score = nli.entailment(pos, "The bridge was built in 1673")
    assert pos_hyp_score > neg_hyp_score


# ----------------------------------------------------------- check 1
def test_retrieval_quality_degree(verifier):
    strong = verifier.check_retrieval_quality([_chunk("a", "x", 0.9), _chunk("b", "y", 0.2)])
    assert strong.detail["degree"] == "correct"
    weak = verifier.check_retrieval_quality([_chunk("a", "x", 0.05)])
    assert weak.detail["degree"] == "incorrect"
    empty = verifier.check_retrieval_quality([])
    assert empty.score == 0.0


# ----------------------------------------------------------- check 2
def test_attribution_separates_supported_and_unsupported(verifier):
    ev = [_chunk("c1", "Kurosawa was a Japanese filmmaker.")]
    claims = [
        Claim("Kurosawa was Japanese", ["c1"]),          # entailed
        Claim("Kurosawa won an Olympic medal", ["c1"]),  # not entailed
    ]
    res = verifier.check_attribution(claims, ev)
    assert "Kurosawa won an Olympic medal" in res.detail["unsupported"]
    assert "Kurosawa was Japanese" not in res.detail["unsupported"]
    assert 0.0 < res.score < 1.0


# ----------------------------------------------------------- check 4
def test_coverage_flags_uncovered_hop(verifier):
    ev = [_chunk("c1", "Ran was directed by Akira Kurosawa.")]
    hops = [
        SubQuery("hop_0", "Who directed Ran?", 0),
        SubQuery("hop_1", "What is the population of Mercury colony?", 1),
    ]
    res = verifier.check_coverage(hops, ev)
    assert "hop_1" in res.detail["failing_hops"]
    assert "hop_0" not in res.detail["failing_hops"]


# ----------------------------------------------------------- check 5
def test_conflict_detects_contradiction(verifier):
    ev = [
        _chunk("c1", "The treaty was signed in 1905 by both nations."),
        _chunk("c2", "The treaty was not signed in 1905 by both nations."),
    ]
    res = verifier.check_conflict(ev)
    assert res.detail["conflict"] is True


# ------------------------------------------------------- aggregation
def _state(question, claims, evidence, sub_queries=None, answerable=True):
    s = RAGState(question=question)
    s.claims = claims
    s.evidence = evidence
    s.answerable = answerable
    s.sub_queries = sub_queries or [SubQuery("hop_0", question, 0)]
    return s


def test_diagnosis_generation_fault(verifier):
    # Evidence clearly present (high score) but the claim isn't entailed by it.
    ev = [_chunk("c1", "Ran is a 1985 samurai film directed by Akira Kurosawa.", 0.9)]
    s = _state(
        "Who directed Ran?",
        [Claim("Ran was directed by Steven Spielberg in Hollywood", ["c1"])],
        ev,
    )
    res = verifier.verify(s)
    assert res.diagnosis == GENERATION_FAULT
    assert res.evidence_seen is True


def test_diagnosis_retrieval_fault_when_nothing_ontopic(verifier):
    ev = [_chunk("c1", "Unrelated text about gardening tools.", 0.02)]
    s = _state("Who directed Ran?", [], ev, answerable=False)
    res = verifier.verify(s)
    assert res.diagnosis == RETRIEVAL_FAULT
    assert res.evidence_seen is False
    assert res.suggested_queries  # offers something to re-retrieve


def test_diagnosis_planning_fault_uncovered_hop(verifier):
    ev = [_chunk("c1", "Ran was directed by Akira Kurosawa.", 0.9)]
    hops = [
        SubQuery("hop_0", "Who directed Ran?", 0),
        SubQuery("hop_1", "What nationality is the Mercury colony governor?", 1),
    ]
    s = _state(
        "What nationality is the director of Ran?",
        [Claim("Ran was directed by Akira Kurosawa", ["c1"])],
        ev,
        sub_queries=hops,
    )
    res = verifier.verify(s)
    assert res.diagnosis == PLANNING_FAULT
    assert "hop_1" in res.failing_hops


def test_diagnosis_conflict(verifier):
    ev = [
        _chunk("c1", "The treaty was signed in 1905 by both delegations.", 0.8),
        _chunk("c2", "The treaty was not signed in 1905 by both delegations.", 0.8),
    ]
    s = _state(
        "When was the treaty signed?",
        [Claim("The treaty was signed in 1905", ["c1"])],
        ev,
    )
    res = verifier.verify(s)
    assert res.diagnosis == CONFLICT
    assert res.conflict is True


def test_diagnosis_pass_on_clean_answer(verifier):
    ev = [_chunk("c1", "Akira Kurosawa was a Japanese filmmaker born in Tokyo.", 0.9)]
    s = _state(
        "What nationality was Akira Kurosawa?",
        [Claim("Akira Kurosawa was Japanese", ["c1"])],
        ev,
    )
    res = verifier.verify(s)
    assert res.diagnosis == PASS
    assert res.confidence > 0.4
    # State is mirrored for downstream stages.
    assert s.diagnosis == PASS and s.confidence == pytest.approx(res.confidence)


def test_self_consistency_optional_and_runs_with_sampler(verifier):
    s = _state("Q?", [Claim("x", ["c1"])], [_chunk("c1", "x", 0.9)])
    # Without sampler the check is skipped.
    res = verifier.verify(s)
    assert "consistency" not in res.checks
    # With a stable sampler agreement is high.
    res2 = verifier.verify(s, sampler=lambda q: "japanese", self_consistency_k=4)
    assert res2.checks["consistency"].score == 1.0
