"""Controller / router + bounded control loop (doc sections 4.7 - 4.8).

This is where the design departs hardest from naive RAG: the loop-back target
depends on the verifier's *diagnosis*, not on a single boolean. The controller
runs the pipeline once, verifies, and then -- while a correction is still worth
its cost -- routes each diagnosis to its targeted correction and re-verifies.

Routing table (doc section 4.7)
  retrieval_fault  -> escalating re-retrieval of the failing hop only
                      (reformulate -> HyDE -> out-of-corpus/web fallback)
  generation_fault -> regenerate with only the relevant chunks (reduce
                      distraction), evidence unchanged
  planning_fault   -> fill the templated failing hop from its prerequisite
                      hop's answer, then re-retrieve that hop
  conflict         -> conflict-resolution generation (recency / authority /
                      majority), evidence unchanged
  pass             -> finalize

Stopping conditions (doc section 4.8, P5) -- terminate on ANY of:
  1. confidence >= tau           (success)
  2. budget exhausted            (max corrections)
  3. no new evidence             (a retrieval correction surfaced only
                                  already-seen chunk ids)

`seen_chunk_ids` is maintained across passes and evidence is accumulated (hop-1
evidence is never discarded when fetching hop-2), so corrections are surgical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .finalizer import Finalizer
from .generator import GroundedGenerator
from .pipeline import Stage1Pipeline
from .state import Claim, RAGState, SubQuery
from .verifier import (
    CONFLICT,
    GENERATION_FAULT,
    PASS,
    PLANNING_FAULT,
    RETRIEVAL_FAULT,
    Verifier,
)

# Optional out-of-corpus fallback: maps a query -> list of {id,text,source,...}.
WebSearch = Callable[[str], list[dict]]

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Budget:
    max_corrections: int = 3
    corrections_used: int = 0

    def exhausted(self) -> bool:
        return self.corrections_used >= self.max_corrections


# Actions that consume retrieval and are therefore subject to the
# no-new-evidence stop. Pure re-synthesis actions are not.
_RETRIEVAL_ACTIONS = {"fill-and-retrieve", "reformulate", "hyde", "web"}


class Controller:
    def __init__(
        self,
        pipeline: Stage1Pipeline,
        verifier: Optional[Verifier] = None,
        *,
        tau: float = 0.55,
        tau_abstain: float = 0.30,
        max_corrections: int = 3,
        web_search: Optional[WebSearch] = None,
    ) -> None:
        self.pipeline = pipeline
        self.verifier = verifier or Verifier()
        self.tau = tau            # tau_answer: loop success + answer threshold
        self.tau_abstain = tau_abstain
        self.max_corrections = max_corrections
        self.web_search = web_search
        self.finalizer = Finalizer(tau_answer=tau, tau_abstain=tau_abstain)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self, question: str) -> RAGState:
        state = self.pipeline.run(question)
        budget = Budget(max_corrections=self.max_corrections)
        retrieval_pass = 0
        # Escalation rungs: 0 reformulate, 1 HyDE, 2 web (only if configured).
        last_rung = 2 if self.web_search is not None else 1

        while True:
            result = self.verifier.verify(state)

            if result.diagnosis == PASS or (
                result.confidence is not None and result.confidence >= self.tau
            ):
                state.log("control", action="finalize", reason="pass",
                          confidence=round(result.confidence or 0.0, 3))
                break
            if budget.exhausted():
                state.log("control", action="finalize", reason="budget_exhausted")
                break

            if result.diagnosis == RETRIEVAL_FAULT:
                action, new_chunks = self._fix_retrieval(state, result, retrieval_pass)
                ladder_exhausted = retrieval_pass >= last_rung
                retrieval_pass += 1
            else:
                action, new_chunks = self._correct(state, result, budget.corrections_used)
                ladder_exhausted = True

            budget.corrections_used += 1
            state.correction_count = budget.corrections_used
            state.new_chunks_last_pass = new_chunks
            state.log("correct", diagnosis=result.diagnosis, action=action,
                      new_chunks=new_chunks, pass_no=budget.corrections_used)

            if action in {"reformulate", "hyde", "web"}:
                # Retrieval escalation: surface new evidence then resynthesize.
                # The no-new-evidence stop fires only once the ladder is
                # exhausted -- a reformulate that finds nothing doesn't mean
                # HyDE or the web fallback will also come up empty.
                if new_chunks > 0:
                    self._resynthesize(state)
                elif ladder_exhausted:
                    state.log("control", action="finalize", reason="no_new_evidence")
                    break
            elif action == "fill-and-retrieve":
                # Filling a templated hop changes the generation target (the now
                # resolved terminal hop), so always resynthesize -- even on a
                # small corpus where the needed chunk was already retrieved.
                self._resynthesize(state)
            # regenerate / resolve-conflict already applied their generation.

        self._finalize(state)
        return state

    # ------------------------------------------------------------------ #
    # Diagnosis -> targeted correction
    # ------------------------------------------------------------------ #
    def _correct(self, state: RAGState, result, pass_idx: int) -> tuple[str, int]:
        if result.diagnosis == PLANNING_FAULT:
            return self._fix_planning(state, result)
        if result.diagnosis == RETRIEVAL_FAULT:
            return self._fix_retrieval(state, result, pass_idx)
        if result.diagnosis == GENERATION_FAULT:
            return self._fix_generation(state, result)
        if result.diagnosis == CONFLICT:
            return self._fix_conflict(state, result)
        return ("none", 0)

    # ---- planning: fill the templated hop, then re-retrieve it -------- #
    def _fix_planning(self, state: RAGState, result) -> tuple[str, int]:
        hop_map = {sq.id: sq for sq in state.sub_queries}
        new_total = 0
        acted = False
        for hop_id in result.failing_hops:
            sq = hop_map.get(hop_id)
            if sq is None:
                continue
            if not sq.resolved and sq.depends_on:
                entity = self._resolve_hop_answer(state, sq.depends_on)
                if not entity:
                    continue
                filled = sq.fill(entity)
                # Replace the sub-query in place (keep ordering).
                idx = next(i for i, s in enumerate(state.sub_queries) if s.id == hop_id)
                state.sub_queries[idx] = filled
                before = {c.id for c in state.evidence}
                new_total += self._retrieve_into(state, filled.text)
                new_ids = [c.id for c in state.evidence if c.id not in before]
                state.log("fill", hop=hop_id, entity=entity,
                          query=filled.text, new_ids=new_ids)
                acted = True
            else:
                # Resolved-but-uncovered hop: re-retrieve with its own phrasing.
                new_total += self._retrieve_into(state, sq.text)
                acted = True
        if not acted:
            # Fall back to treating it as a retrieval fault.
            return self._fix_retrieval(state, result, 0)
        return ("fill-and-retrieve", new_total)

    # ---- retrieval: escalating re-retrieval of the failing hop -------- #
    def _fix_retrieval(self, state: RAGState, result, pass_idx: int) -> tuple[str, int]:
        queries = result.suggested_queries or [state.question]
        # Escalation ladder: reformulate -> HyDE -> web/out-of-corpus.
        if pass_idx == 0:
            new = sum(self._retrieve_into(state, q, k_boost=1.5) for q in queries)
            return ("reformulate", new)
        if pass_idx == 1:
            new = 0
            for q in queries:
                hypo = self._hyde_passage(state, q)
                new += self._retrieve_into(state, hypo)
            return ("hyde", new)
        # Pass 2+: out-of-corpus fallback if available (CRAG-style).
        if self.web_search is not None:
            new = sum(self._web_into(state, q) for q in queries)
            return ("web", new)
        # No web fallback configured -> nothing new can be surfaced.
        return ("web", 0)

    # ---- generation: regenerate with only the relevant chunks --------- #
    def _fix_generation(self, state: RAGState, result) -> tuple[str, int]:
        # Reduce distraction: keep only chunks cited by supported claims plus the
        # single top-reranked chunk, then regenerate under the strict contract.
        cited = {cid for c in state.claims for cid in c.citations}
        relevant = [c for c in state.evidence if c.id in cited]
        if not relevant:
            relevant = state.evidence[:3]
        elif state.evidence:
            top = state.evidence[0]
            if all(c.id != top.id for c in relevant):
                relevant = [top] + relevant
        gen = self.pipeline.generator.generate(state.question, relevant)
        self._apply_generation(state, gen)
        return ("regenerate", 0)

    # ---- conflict: resolve by recency / authority / majority ---------- #
    def _fix_conflict(self, state: RAGState, result) -> tuple[str, int]:
        # Find the two conflicting chunks flagged by the verifier (or top two).
        con = self.verifier.check_conflict(state.evidence)
        between = con.detail.get("between", [])
        by_id = {c.id: c for c in state.evidence}
        pair = [by_id[i] for i in between if i in by_id] or state.evidence[:2]
        if not pair:
            return ("resolve-conflict", 0)
        # Resolution policy: most recent timestamp wins; ties -> keep both and
        # note the discrepancy.
        winner = max(pair, key=lambda c: (c.timestamp or "", c.rerank_score))
        gen = self.pipeline.generator.generate(state.question, [winner])
        note = (
            f"Sources disagree ({', '.join(c.id for c in pair)}); resolved by "
            f"recency to {winner.id}"
            + (f" (dated {winner.timestamp})" if winner.timestamp else "")
            + "."
        )
        gen.explanation = (note + " " + gen.explanation).strip()
        self._apply_generation(state, gen)
        state.conflict = True
        return ("resolve-conflict", 0)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _retrieve_into(self, state: RAGState, query: str, k_boost: float = 1.0) -> int:
        """Retrieve `query`, add only unseen chunks, return how many were new."""
        k = int(self.pipeline.retrieve_k * k_boost)
        candidates = self.pipeline.retriever.retrieve(query, top_k=k)
        new = [c for c in candidates if c.id not in state.seen_chunk_ids]
        for c in new:
            state.seen_chunk_ids.add(c.id)
        state.evidence.extend(new)
        return len(new)

    def _web_into(self, state: RAGState, query: str) -> int:
        if self.web_search is None:
            return 0
        from .state import Chunk
        added = 0
        for d in self.web_search(query) or []:
            cid = d.get("id") or f"web::{abs(hash(d.get('text', ''))) % 10**8}"
            if cid in state.seen_chunk_ids:
                continue
            chunk = Chunk(
                id=cid, text=d.get("text", ""), source=d.get("source", "web"),
                timestamp=d.get("timestamp", ""),
            )
            chunk.rerank_score = float(d.get("rerank_score", 0.5))
            state.seen_chunk_ids.add(cid)
            state.evidence.append(chunk)
            added += 1
        return added

    def _hyde_passage(self, state: RAGState, query: str) -> str:
        """HyDE: a hypothetical answer passage to retrieve against.

        With the offline extractive generator this degrades to query expansion
        over current evidence; with a real LLM generator it produces a synthetic
        passage that embeds near real answer docs.
        """
        gen = self.pipeline.generator.generate(query, state.evidence[:5])
        hypo = (gen.answer + ". " + gen.explanation).strip(". ")
        return f"{query} {hypo}".strip()

    def _resolve_hop_answer(self, state: RAGState, hop_id: str) -> str:
        """Answer a prerequisite hop from current evidence (for bridge fills)."""
        sq = next((s for s in state.sub_queries if s.id == hop_id), None)
        if sq is None:
            return ""
        gen = self.pipeline.generator.generate(sq.text, state.evidence)
        return gen.answer.strip()

    def _resynthesize(self, state: RAGState) -> None:
        """Re-rank accumulated evidence and regenerate the answer.

        For a bridge, the information need of the *final* hop is what the answer
        should report, so generation targets the resolved terminal hop when one
        exists; otherwise the original question.
        """
        # Re-rank against the question we are about to ANSWER (the resolved
        # terminal hop for a bridge), not the original phrasing -- otherwise the
        # freshly-retrieved answer passage can be ranked out before we use it.
        gen_query = self._generation_query(state)
        reranked = self.pipeline.reranker.rerank(
            gen_query, state.evidence, top_n=self.pipeline.rerank_top_n
        )
        state.evidence = reranked
        report = self.pipeline.reranker.score_report(reranked)
        state.log("rerank", kept=len(reranked), ids=[c.id for c in reranked], **report)

        gen = self.pipeline.generator.generate(gen_query, reranked)
        # Carry forward the bridge claim from the prerequisite hop so the final
        # answer remains fully attributed.
        if gen_query != state.question:
            gen.claims = self._bridge_claims(state) + gen.claims
        self._apply_generation(state, gen)

    def _generation_query(self, state: RAGState) -> str:
        terminal = [
            s for s in state.sub_queries
            if s.resolved and s.depends_on and s.template
        ]
        if terminal:
            return max(terminal, key=lambda s: s.hop_index).text
        return state.question

    def _bridge_claims(self, state: RAGState) -> list[Claim]:
        """Pull a supporting claim for each resolved prerequisite hop."""
        claims: list[Claim] = []
        prereq_ids = {s.depends_on for s in state.sub_queries if s.depends_on}
        for hop_id in prereq_ids:
            sq = next((s for s in state.sub_queries if s.id == hop_id), None)
            if sq is None:
                continue
            g = self.pipeline.generator.generate(sq.text, state.evidence)
            claims.extend(g.claims[:1])
        return claims

    def _apply_generation(self, state: RAGState, gen) -> None:
        state.answer = gen.answer
        state.claims = _dedup_claims(gen.claims)
        state.answerable = gen.answerable
        state.explanation = gen.explanation
        state.unsupported_claims = [c.text for c in gen.claims if not c.supported]
        state.citations = sorted({cid for c in gen.claims for cid in c.citations})

    def _finalize(self, state: RAGState) -> None:
        # Calibrated abstention (doc section 4.9): answered / hedged / abstained
        # based on the verifier's confidence vs. tau_answer / tau_abstain.
        self.finalizer.finalize(state)


def _dedup_claims(claims):
    """Drop duplicate claims (same text + citations) while preserving order."""
    seen = set()
    out = []
    for c in claims:
        key = (c.text.strip().lower(), tuple(sorted(c.citations)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
