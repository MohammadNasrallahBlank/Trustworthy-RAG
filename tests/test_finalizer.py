"""Tests for the Stage 5 finalizer (calibrated abstention, doc section 4.9)."""

from __future__ import annotations

from srag import ANSWERED, ABSTAINED, HEDGED, Claim, Finalizer, RAGState, SubQuery


def _state(confidence, answer="Japanese", claims=None, answerable=True,
           failing_hops=None, unsupported=None, sub_queries=None):
    s = RAGState(question="What nationality is the director of Ran?")
    s.confidence = confidence
    s.answer = answer
    s.answerable = answerable
    s.claims = claims if claims is not None else [Claim("X is Y", ["c1"])]
    s.failing_hops = failing_hops or []
    s.unsupported_claims = unsupported or []
    s.sub_queries = sub_queries or [SubQuery("hop_0", "q", 0)]
    return s


def test_answered_above_tau_answer():
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    s = f.finalize(_state(0.80))
    assert s.answer_status == ANSWERED
    assert s.abstained is False
    assert s.final_answer == "Japanese"
    assert "Japanese" in f.message(s)


def test_hedged_in_middle_band():
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    s = f.finalize(_state(0.42, failing_hops=["hop_1"],
                          sub_queries=[SubQuery("hop_1", "What nationality is X?", 1)]))
    assert s.answer_status == HEDGED
    assert s.abstained is False
    assert s.final_answer == "Japanese"           # best partial retained
    assert s.missing                               # what could not be verified
    assert "Likely" in f.message(s)
    assert "could not verify" in f.message(s).lower()


def test_mid_confidence_with_an_answer_hedges_not_abstains():
    # In the hedge band, surface the best answer with caveats -- don't veto it.
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    s = f.finalize(_state(0.42, unsupported=["X is Y"]))
    assert s.answer_status == HEDGED
    assert s.abstained is False
    assert s.final_answer == "Japanese"


def test_very_low_confidence_abstains_even_with_a_span():
    # If confidence is below tau_abstain the evidence isn't there -> refuse,
    # even if the generator emitted a (likely spurious) answer span.
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    s = f.finalize(_state(0.10))
    assert s.answer_status == ABSTAINED
    assert s.abstained is True


def test_abstains_only_when_no_answer():
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    s = f.finalize(_state(0.10, answer="", claims=[], answerable=False))
    assert s.answer_status == ABSTAINED
    assert s.abstained is True
    assert s.final_answer == ""
    assert "couldn't find reliable evidence" in f.message(s).lower()


def test_hard_unanswerable_abstains_regardless_of_confidence():
    f = Finalizer(tau_answer=0.55, tau_abstain=0.30)
    # No claims and generation said unanswerable -> abstain even if conf is high.
    s = f.finalize(_state(0.99, answer="", claims=[], answerable=False))
    assert s.answer_status == ABSTAINED
    assert s.abstained is True


def test_missing_lists_failing_hops_and_unsupported():
    f = Finalizer()
    s = _state(0.42, failing_hops=["hop_1"], unsupported=["Z claim"],
               sub_queries=[SubQuery("hop_1", "What nationality is Kurosawa?", 1)])
    f.finalize(s)
    joined = " ".join(s.missing)
    assert "What nationality is Kurosawa?" in joined
    assert "Z claim" in joined
