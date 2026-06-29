"""Finalizer & calibrated abstention (doc section 4.9).

Abstention is a first-class output: a well-placed "I couldn't verify this" beats
a confident hallucination. But a *working* system should also give its best
answer when it has one -- refusing a question you can actually answer is its own
failure. So the finalizer:

  * ABSTAINS only when there is genuinely nothing to offer -- no answer span, or
    confidence so low the evidence clearly isn't there (a real unanswerable);
  * ANSWERS when confident (>= tau_answer);
  * HEDGES otherwise -- surfaces the grounded best-guess with an explicit note of
    what couldn't be verified, rather than vetoing its own answer.

Two baseline modes (off by default) support the trust eval (plan v2 section 3.1):
  * always_answer -> never abstain on confidence/coverage; answer whenever a span
    exists. Powers `plain_rag_always_answer`, and carries `prompted_abstain_rag`
    (whose abstention comes from the generator marking answerable=False).
  * score_only -> abstain PURELY on confidence < tau_abstain, ignoring the
    answer/coverage logic. Powers `retrieval_score_abstain`.

Thresholds default to sensible values; `srag.calibration.calibrate_thresholds`
fits them on a dev set with a known-unanswerable subset (stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import RAGState

ANSWERED = "answered"
HEDGED = "hedged"
ABSTAINED = "abstained"


@dataclass
class Finalizer:
    tau_answer: float = 0.55
    tau_abstain: float = 0.30
    always_answer: bool = False
    score_only: bool = False

    def finalize(self, state: RAGState) -> RAGState:
        """Set final_answer / answer_status / abstained / missing on the state."""
        conf = state.confidence if state.confidence is not None else 0.0
        missing = self._missing(state)

        no_answer = (not state.answer.strip()) or (
            (not state.answerable) and not state.claims
        )

        if self.always_answer:
            # Answer whenever there is a span; abstain only when there is nothing
            # at all to return (the generator produced no answer).
            if no_answer:
                status, final, abstained = ABSTAINED, "", True
            else:
                status, final, abstained = ANSWERED, state.answer, False
        elif self.score_only:
            # Pure confidence gate -- no coverage/answerability reasoning.
            if conf < self.tau_abstain:
                status, final, abstained = ABSTAINED, "", True
            else:
                status, final, abstained = ANSWERED, state.answer, False
        elif no_answer or conf < self.tau_abstain:
            # Nothing to offer, or confidence is so low the evidence isn't there
            # (e.g. a genuinely unanswerable question) -> refuse.
            status = ABSTAINED
            final = ""
            abstained = True
        elif conf >= self.tau_answer:
            status = ANSWERED
            final = state.answer
            abstained = False
        else:
            # Has a grounded best-guess but isn't fully sure -> hedge, don't veto.
            status = HEDGED
            final = state.answer
            abstained = False

        state.answer_status = status
        state.final_answer = final
        state.abstained = abstained
        state.missing = missing
        if status != ABSTAINED:
            state.citations = sorted(
                {cid for c in state.claims for cid in c.citations}
            )
        state.log(
            "finalize",
            status=status,
            confidence=round(conf, 3),
            answer=final,
            missing=missing,
            citations=state.citations if status != ABSTAINED else [],
        )
        return state

    def message(self, state: RAGState) -> str:
        """A human-readable rendering of the finalized outcome."""
        if state.answer_status == ANSWERED:
            cites = ", ".join(state.citations)
            return f"{state.final_answer}" + (f" [{cites}]" if cites else "")
        if state.answer_status == HEDGED:
            miss = "; ".join(state.missing) or "some details"
            cites = ", ".join(state.citations)
            return (
                f"Likely: {state.final_answer}"
                + (f" [{cites}]" if cites else "")
                + f" -- but I could not verify: {miss}."
            )
        # ABSTAINED
        miss = "; ".join(state.missing) or "the required evidence"
        return (
            "I couldn't find reliable evidence to answer this. "
            f"Missing: {miss}."
        )

    # ------------------------------------------------------------------ #
    def _missing(self, state: RAGState) -> list[str]:
        out: list[str] = []
        hop_map = {sq.id: sq for sq in state.sub_queries}
        for hop_id in state.failing_hops:
            sq = hop_map.get(hop_id)
            out.append(f"evidence for: {sq.text}" if sq else f"hop {hop_id}")
        out.extend(f"unverified claim: {c}" for c in state.unsupported_claims)
        if not out and not state.answerable:
            out.append("no on-topic evidence was retrieved")
        seen: set[str] = set()
        uniq = []
        for m in out:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq
