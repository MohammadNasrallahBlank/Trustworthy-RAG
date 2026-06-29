"""System configurations for the accuracy-vs-cost curve (doc section 7).

Each config is a `Runner` (name + `run(question) -> RAGState`) built over the
same indexed corpus, so the curve compares like with like:

  naive               single DENSE pass, no reranker, no loop.
  reranker_baseline   hybrid (BM25+dense, RRF) + cross-encoder rerank, single
                      pass -- THE BAR the full loop must beat.
  planning            reranker_baseline + multi-hop planner (still single pass).
  full                planning + verification + targeted correction + abstention.
  full_no_planning    ablation: full loop without the planner.
  full_no_correction  ablation: verify + abstain but max_corrections = 0.

All components use the deterministic offline fallbacks so the curve is
reproducible without model weights; swap in real models by passing
`prefer_fallback=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..controller import Controller
from ..embeddings import Embedder
from ..entailment import EntailmentModel
from ..finalizer import Finalizer
from ..generator import GroundedGenerator
from ..planner import Planner
from ..reranker import CrossEncoderReranker
from ..llm import make_llm_generator_fn, make_llm_planner_fn
from ..verifier import Verifier
from ..retrieval import HybridRetriever
from ..state import Chunk, RAGState
from ..pipeline import Stage1Pipeline

CONFIGS = [
    "naive",
    "reranker_baseline",
    "planning",
    "full",
    "full_no_planning",
    "full_no_correction",
]


@dataclass
class Runner:
    name: str
    run: Callable[[str], RAGState]


def _pipeline(documents, *, fusion, use_reranker, planner, prefer_fallback,
              pre_chunked=False, chat=None, llm_planner=True):
    # With a real chat LLM, use it for grounded generation and (if this config
    # plans) for query decomposition; otherwise the deterministic offline paths.
    generator = GroundedGenerator(make_llm_generator_fn(chat)) if chat else GroundedGenerator()
    if chat is not None and planner is not None and llm_planner:
        planner = Planner(llm=make_llm_planner_fn(chat))
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=prefer_fallback)),
        reranker=CrossEncoderReranker(prefer_fallback=prefer_fallback),
        generator=generator,
        planner=planner,
        retrieval_fusion=fusion,
        use_reranker=use_reranker,
    )
    if pre_chunked:
        # Index documents as atomic chunks, preserving their ids exactly (so
        # gold supporting-chunk ids line up). Used for HotpotQA-style data where
        # each context sentence is already a unit.
        chunks = [
            Chunk(id=d["id"], text=d["text"], source=d.get("source", ""))
            for d in documents
        ]
        return pipe.index_chunks(chunks)
    return pipe.index_documents(documents)


def _pipeline_runner(pipe, name) -> Runner:
    # No verifier in these configs: abstain only when generation found nothing.
    def run(q):
        s = pipe.run(q)
        s.final_answer = s.answer
        s.answer_status = "answered" if s.answerable else "abstained"
        s.abstained = not s.answerable
        return s
    return Runner(name, run)


def _controller_runner(pipe, name, *, thresholds, max_corrections,
                       prefer_fallback, entailment=None) -> Runner:
    ta, tab = thresholds if thresholds else (0.55, 0.30)
    ent = entailment or EntailmentModel(prefer_fallback=prefer_fallback)
    ctrl = Controller(
        pipe,
        verifier=Verifier(ent),
        tau=ta,
        tau_abstain=tab,
        max_corrections=max_corrections,
    )
    return Runner(name, ctrl.run)


def build_config(name: str, documents: Sequence[dict], *,
                 prefer_fallback: bool = True,
                 thresholds: Optional[tuple] = None,
                 pre_chunked: bool = False,
                 chat=None,
                 llm_planner: bool = True,
                 entailment=None,
                 max_corrections: int = 3) -> Runner:
    if name == "naive":
        pipe = _pipeline(documents, fusion="dense", use_reranker=False,
                         planner=None, prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _pipeline_runner(pipe, name)
    if name == "reranker_baseline":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True,
                         planner=None, prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _pipeline_runner(pipe, name)
    if name == "planning":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True,
                         planner=Planner(), prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _pipeline_runner(pipe, name)
    if name == "full":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True,
                         planner=Planner(), prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _controller_runner(pipe, name, thresholds=thresholds,
                                  max_corrections=max_corrections,
                                  prefer_fallback=prefer_fallback,
                                  entailment=entailment)
    if name == "full_no_planning":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True,
                         planner=None, prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _controller_runner(pipe, name, thresholds=thresholds,
                                  max_corrections=max_corrections,
                                  prefer_fallback=prefer_fallback,
                                  entailment=entailment)
    if name == "full_no_correction":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True,
                         planner=Planner(), prefer_fallback=prefer_fallback, pre_chunked=pre_chunked, chat=chat,
                         llm_planner=llm_planner)
        return _controller_runner(pipe, name, thresholds=thresholds,
                                  max_corrections=0, prefer_fallback=prefer_fallback,
                                  entailment=entailment)
    raise ValueError(f"unknown config: {name!r} (known: {CONFIGS})")


# ====================================================================== #
# Trustworthy-RAG baselines (plan v2 section 3.1)
# ====================================================================== #
# Five systems sharing the SAME retriever (hybrid RRF), reranker (cross-encoder),
# and LLM. They differ only in the abstention policy, isolating the variable the
# project is about: producing fewer unsupported confident answers (UCA) while
# preserving answerable coverage.
#
#   plain_rag_always_answer  retrieve->generate, ALWAYS answers (no abstain).
#   prompted_abstain_rag     same, but the generator is prompted to say IDK when
#                            the context lacks the answer (one-line prompt change).
#   retrieval_score_abstain  abstain purely on retrieval confidence; no verifier.
#   guarded                  verify (claim-NLI + coverage + retrieval) + calibrated
#                            abstention; NO correction loop.
#   guarded+correct          `guarded` plus the bounded self-correcting loop.
#
# `guarded` and `guarded+correct` are identical except max_corrections (0 vs N),
# so any difference between them is attributable to the loop alone.

TRUST_CONFIGS = [
    "plain_rag_always_answer",
    "prompted_abstain_rag",
    "retrieval_score_abstain",
    "guarded",
    "guarded+correct",
]


def _finalizer_runner(pipe, name, finalizer: Finalizer, *, score_fn=None) -> Runner:
    """Run the single-pass pipeline, then apply a baseline finalizer.

    `score_fn(state) -> float` optionally sets state.confidence before finalizing
    (used by `retrieval_score_abstain`, which gates on retrieval confidence).
    """
    def run(q):
        s = pipe.run(q)
        if score_fn is not None:
            s.confidence = score_fn(s)
        finalizer.finalize(s)
        return s
    return Runner(name, run)


def _index(pipe, documents, pre_chunked):
    if pre_chunked:
        return pipe.index_chunks([
            Chunk(id=d["id"], text=d["text"], source=d.get("source", ""))
            for d in documents
        ])
    return pipe.index_documents(documents)


def build_trust_config(name: str, documents: Sequence[dict], *,
                       prefer_fallback: bool = True,
                       thresholds: Optional[tuple] = None,
                       pre_chunked: bool = False,
                       chat=None,
                       entailment=None,
                       max_corrections: int = 2) -> Runner:
    ta, tab = thresholds if thresholds else (0.55, 0.30)

    if name == "plain_rag_always_answer":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True, planner=None,
                         prefer_fallback=prefer_fallback, pre_chunked=pre_chunked,
                         chat=chat, llm_planner=False)
        return _finalizer_runner(pipe, name, Finalizer(always_answer=True))

    if name == "prompted_abstain_rag":
        # One-line prompt change: ask the model to refuse when unsupported. The
        # generator's prompt_mode="abstain" drives the IDK; the finalizer just
        # surfaces it (empty span -> abstained).
        gen = (GroundedGenerator(make_llm_generator_fn(chat, prompt_mode="abstain"),
                                 prompt_mode="abstain")
               if chat else GroundedGenerator(prompt_mode="abstain"))
        pipe = Stage1Pipeline(
            retriever=HybridRetriever(Embedder(prefer_fallback=prefer_fallback)),
            reranker=CrossEncoderReranker(prefer_fallback=prefer_fallback),
            generator=gen, planner=None, retrieval_fusion="rrf", use_reranker=True,
        )
        pipe = _index(pipe, documents, pre_chunked)
        return _finalizer_runner(pipe, name, Finalizer(always_answer=True))

    if name == "retrieval_score_abstain":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True, planner=None,
                         prefer_fallback=prefer_fallback, pre_chunked=pre_chunked,
                         chat=chat, llm_planner=False)
        scorer = Verifier(EntailmentModel(prefer_fallback=prefer_fallback))

        def retrieval_score(s):
            return scorer.check_retrieval_quality(s.evidence).score

        return _finalizer_runner(pipe, name,
                                 Finalizer(score_only=True, tau_abstain=tab),
                                 score_fn=retrieval_score)

    if name == "guarded":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True, planner=None,
                         prefer_fallback=prefer_fallback, pre_chunked=pre_chunked,
                         chat=chat, llm_planner=False)
        return _controller_runner(pipe, name, thresholds=(ta, tab),
                                  max_corrections=0, prefer_fallback=prefer_fallback,
                                  entailment=entailment)

    if name == "guarded+correct":
        pipe = _pipeline(documents, fusion="rrf", use_reranker=True, planner=None,
                         prefer_fallback=prefer_fallback, pre_chunked=pre_chunked,
                         chat=chat, llm_planner=False)
        return _controller_runner(pipe, name, thresholds=(ta, tab),
                                  max_corrections=max_corrections,
                                  prefer_fallback=prefer_fallback,
                                  entailment=entailment)

    raise ValueError(f"unknown trust config: {name!r} (known: {TRUST_CONFIGS})")
