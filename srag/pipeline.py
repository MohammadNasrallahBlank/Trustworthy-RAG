"""Pipeline: plan -> chunk-time index -> hybrid retrieve -> rerank -> generate.

With `planner=None` this is the Stage-1 "strong static pipeline" (doc section 8,
step 1) — the single-hop baseline the correction loop must beat. Pass a
`Planner` to enable Stage-3 multi-hop decomposition (the "+planning"
configuration in the eval curve). There is still no controller or abstention
loop here; those are stages 4-5.

Every decision is logged into `RAGState.trace`.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .chunking import chunk_corpus
from .embeddings import Embedder
from .generator import GroundedGenerator
from .gate import RetrievalGate
from .planner import Planner
from .reranker import CrossEncoderReranker
from .retrieval import HybridRetriever
from .state import Chunk, RAGState, SubQuery


class Stage1Pipeline:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[GroundedGenerator] = None,
        planner: Optional[Planner] = None,
        gate: Optional[RetrievalGate] = None,
        *,
        retrieve_k: int = 50,
        rerank_top_n: int = 6,
        retrieval_fusion: str = "rrf",
        use_reranker: bool = True,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroundedGenerator()
        # Eval knobs: "dense"+no reranker == the naive baseline; "rrf"+reranker
        # == the strong hybrid+reranker baseline (the bar the loop must beat).
        self.retrieval_fusion = retrieval_fusion
        self.use_reranker = use_reranker
        # planner=None keeps the Stage-1 single-hop baseline (the bar to beat).
        # Pass a Planner to enable multi-hop decomposition (the "+planning"
        # configuration in the eval curve).
        self.planner = planner
        # Optional active-retrieval gate (stage 6b, doc 4.1).
        self.gate = gate
        self.retrieve_k = retrieve_k
        self.rerank_top_n = rerank_top_n
        self._indexed = False

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_documents(self, documents: Sequence[dict], **chunk_kwargs) -> "Stage1Pipeline":
        """Chunk raw documents and build the retrieval index."""
        chunks = chunk_corpus(documents, **chunk_kwargs)
        return self.index_chunks(chunks)

    def index_chunks(self, chunks: Sequence[Chunk]) -> "Stage1Pipeline":
        self.retriever.index(list(chunks))
        self._indexed = True
        return self

    @property
    def backends(self) -> dict:
        return {
            "embedder": self.retriever.embedder.backend,
            "reranker": self.reranker.backend,
            "generator": "llm" if self.generator.llm else "extractive-offline",
            "planner": (self.planner.backend if self.planner else "single-hop"),
        }

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def run(self, question: str, question_type: Optional[str] = None) -> RAGState:
        if not self._indexed:
            raise RuntimeError("Pipeline.run called before indexing a corpus.")

        state = RAGState(question=question)

        # Plan: decompose into sub-queries (or a single hop if no planner).
        if self.planner is not None:
            qtype, sub_queries = self.planner.plan(question)
            state.question_type = question_type or qtype
            state.sub_queries = sub_queries
        else:
            state.question_type = question_type or "single-hop"
            state.sub_queries = [SubQuery(id="hop_0", text=question, hop_index=0)]
        state.log(
            "plan",
            question_type=state.question_type,
            sub_queries=[s.text for s in state.sub_queries],
            unresolved_hops=[s.id for s in state.sub_queries if not s.resolved],
        )

        # Active-retrieval gate (optional): skip retrieval for queries the model
        # can answer parametrically, then answer directly. The verifier (if run
        # downstream) still checks the parametric answer.
        if self.gate is not None:
            decision = self.gate.should_retrieve(question)
            state.log("gate", retrieve=decision.retrieve, reason=decision.reason,
                      **decision.signals)
            if not decision.retrieve:
                gen = self.generator.generate(question, [])
                state.answer = gen.answer
                state.claims = gen.claims
                state.answerable = gen.answerable
                state.explanation = gen.explanation
                state.unsupported_claims = [c.text for c in gen.claims if not c.supported]
                state.citations = sorted({cid for c in gen.claims for cid in c.citations})
                state.log("generate", answer=gen.answer, n_claims=len(gen.claims),
                          unsupported=len(gen.unsupported_facts),
                          answerable=gen.answerable, parametric=True)
                state.final_answer = gen.answer
                return state

        # Retrieve per hop (accumulate evidence + dedup across hops).
        for sq in state.sub_queries:
            candidates = self.retriever.retrieve(
                sq.text, top_k=self.retrieve_k, fusion=self.retrieval_fusion
            )
            new = [c for c in candidates if c.id not in state.seen_chunk_ids]
            for c in new:
                state.seen_chunk_ids.add(c.id)
            state.evidence.extend(new)
            state.log("retrieve", hop=sq.id, candidates=len(candidates),
                      new_chunks=len(new), new_ids=[c.id for c in new])
        state.new_chunks_last_pass = len(state.evidence)

        # Rerank the accumulated candidate set to the top-N (or, for the naive
        # baseline, skip the cross-encoder and keep the fused order).
        if self.use_reranker:
            reranked = self.reranker.rerank(question, state.evidence, top_n=self.rerank_top_n)
        else:
            reranked = sorted(state.evidence, key=lambda c: -c.fusion_score)[: self.rerank_top_n]
            for c in reranked:
                c.rerank_score = c.fusion_score
        state.evidence = reranked
        report = self.reranker.score_report(reranked)
        state.log("rerank", kept=len(reranked), reranked=self.use_reranker,
                  ids=[c.id for c in reranked], **report)

        # Grounded generation with claim-level citations + extract-first schema.
        gen = self.generator.generate(question, reranked)
        state.answer = gen.answer
        state.claims = gen.claims
        state.answerable = gen.answerable
        state.explanation = gen.explanation
        state.unsupported_claims = [c.text for c in gen.claims if not c.supported]
        state.citations = sorted({cid for c in gen.claims for cid in c.citations})
        state.log(
            "generate",
            answer=gen.answer,
            n_claims=len(gen.claims),
            unsupported=len(gen.unsupported_facts),
            answerable=gen.answerable,
        )

        # No correction loop / abstention yet -> finalize directly.
        state.final_answer = gen.answer
        return state
