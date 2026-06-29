"""The one-import, use-it-on-your-own-data API.

    from srag import SelfCorrectingRAG

    rag = SelfCorrectingRAG.from_documents(my_docs)     # your data
    result = rag.ask("your question")
    print(result.status, "->", result.message)          # answered / hedged / abstained
    for c in result.citations: print("  cited:", c)

`my_docs` is a list of strings, or dicts with {"text", optional "id"/"source"}.
By default it runs on deterministic offline components (no weights, no network) so
you can try it instantly. For production, bring a real LLM and turn on real
encoders:

    from srag import TransformersChat
    rag = SelfCorrectingRAG.from_documents(my_docs,
                                           llm=TransformersChat("Qwen/Qwen2.5-3B-Instruct"),
                                           real_models=True)

Bring your *own* LLM by passing any callable `chat(prompt: str) -> str` (an
OpenAI/Anthropic/vLLM wrapper, etc.) as `llm=`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .controller import Controller
from .embeddings import Embedder
from .entailment import EntailmentModel
from .generator import GroundedGenerator
from .llm import make_llm_generator_fn, make_llm_planner_fn
from .planner import Planner
from .reranker import CrossEncoderReranker
from .retrieval import HybridRetriever
from .pipeline import Stage1Pipeline
from .state import Chunk, RAGState
from .verifier import Verifier
from .calibration import calibrate_thresholds, collect_points


@dataclass
class Answer:
    """The result of `SelfCorrectingRAG.ask`."""

    question: str
    answer: str                 # "" when abstained
    status: str                 # "answered" | "hedged" | "abstained"
    message: str                # human-readable rendering (answer, hedge, or refusal)
    citations: list             # chunk ids backing the answer
    corrections: int            # how many correction passes fired
    diagnoses: list             # the verifier diagnosis at each pass
    abstained: bool
    state: RAGState             # full state + decision trace, if you want the internals

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "status": self.status,
            "message": self.message, "citations": self.citations,
            "corrections": self.corrections, "diagnoses": self.diagnoses,
            "abstained": self.abstained,
        }

    def trace_html(self) -> str:
        from .trace_view import render_trace_html
        return render_trace_html(self.state)

    def trace_markdown(self) -> str:
        from .trace_view import render_trace_markdown
        return render_trace_markdown(self.state)

    def trace_json(self) -> dict:
        """A documented, machine-readable export of the answer + decision trace.

        Schema (see docs/TRACE_FORMAT.md):
          {
            "question": str, "answer": str, "status": str, "abstained": bool,
            "citations": [str], "corrections": int, "diagnoses": [str],
            "confidence": float | None,
            "evidence": [{"id": str, "source": str, "rerank_score": float}],
            "trace": [ {"event": str, ...} ]   # ordered pipeline events
          }
        """
        st = self.state
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "abstained": self.abstained,
            "citations": list(self.citations),
            "corrections": self.corrections,
            "diagnoses": list(self.diagnoses),
            "confidence": getattr(st, "confidence", None),
            "evidence": [
                {"id": c.id, "source": getattr(c, "source", ""),
                 "rerank_score": round(float(getattr(c, "rerank_score", 0.0)), 4)}
                for c in (getattr(st, "evidence", []) or [])
            ],
            "trace": list(getattr(st, "trace", []) or []),
        }

    def __repr__(self) -> str:
        return f"Answer(status={self.status!r}, answer={self.answer!r}, corrections={self.corrections})"


def _normalize_docs(documents: Sequence[Any]) -> list[dict]:
    out: list[dict] = []
    for i, d in enumerate(documents):
        if isinstance(d, str):
            out.append({"id": f"doc{i}", "text": d, "source": ""})
        elif isinstance(d, dict):
            out.append({
                "id": str(d.get("id", f"doc{i}")),
                "text": d.get("text", d.get("content", "")),
                "source": d.get("source", d.get("title", "")),
            })
        else:
            raise TypeError(f"document {i} must be str or dict, got {type(d).__name__}")
    return out


class SelfCorrectingRAG:
    """A ready-to-use self-correcting RAG over your documents.

    Verifies each answer against the retrieved evidence, diagnoses the failure
    mode, applies a targeted correction, and abstains (calibrated) instead of
    hallucinating — returning grounded answers with citations.
    """

    def __init__(self, controller: Controller):
        self.controller = controller
        self.mode = "self_correcting" if controller.max_corrections else "trustworthy" 

    # ------------------------------------------------------------------ #
    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Any],
        *,
        llm: Optional[Callable[[str], str]] = None,
        real_models: bool = False,
        mode: str = "trustworthy",
        use_planner: bool = True,
        max_corrections: int = 2,
        tau_answer: float = 0.55,
        tau_abstain: float = 0.30,
        pre_chunked: bool = False,
        **chunk_kwargs,
    ) -> "SelfCorrectingRAG":
        """Build a self-correcting RAG over `documents`.

        Args:
            documents: list of strings or {"text", "id"?, "source"?} dicts.
            llm: any `chat(prompt)->str` callable (your model). If None, uses the
                deterministic offline extractive generator (great for trying it).
            real_models: load real sentence-transformers + NLI (else offline
                fallbacks). Independent of `llm`.
            mode: "trustworthy" (default) = verify + calibrated abstention +
                citations, NO correction loop -- the recommended, honest default.
                "self_correcting" = additionally run the bounded diagnose->correct
                loop (costs more retrieval/generation per answer).
            use_planner: enable multi-hop query decomposition.
            max_corrections / tau_answer / tau_abstain: loop + abstention knobs.
            pre_chunked: treat documents as atomic chunks (keep ids as-is).
        """
        prefer_fallback = not real_models
        embedder = Embedder(prefer_fallback=prefer_fallback)
        reranker = CrossEncoderReranker(prefer_fallback=prefer_fallback)
        entailment = EntailmentModel(prefer_fallback=prefer_fallback)
        generator = GroundedGenerator(make_llm_generator_fn(llm)) if llm else GroundedGenerator()
        planner = None
        if use_planner:
            planner = Planner(llm=make_llm_planner_fn(llm)) if llm else Planner()

        pipe = Stage1Pipeline(
            retriever=HybridRetriever(embedder), reranker=reranker,
            generator=generator, planner=planner,
        )
        docs = _normalize_docs(documents)
        if pre_chunked:
            pipe.index_chunks([Chunk(id=d["id"], text=d["text"], source=d["source"]) for d in docs])
        else:
            pipe.index_documents(docs, **chunk_kwargs)

        if mode not in ("trustworthy", "self_correcting"):
            raise ValueError(
                f"mode must be 'trustworthy' or 'self_correcting', got {mode!r}")
        effective_corrections = 0 if mode == "trustworthy" else max_corrections

        ctrl = Controller(pipe, Verifier(entailment), tau=tau_answer,
                          tau_abstain=tau_abstain,
                          max_corrections=effective_corrections)
        obj = cls(ctrl)
        obj.mode = mode
        return obj

    # ------------------------------------------------------------------ #
    def calibrate(self, dev_set: Sequence[dict]) -> "SelfCorrectingRAG":
        """Tune abstention thresholds on labelled examples.

        `dev_set` is a list of {"question": str, "answerable": bool}. Mix a few
        questions your corpus *can* answer with a few it *can't*; this fits when
        to answer vs. abstain.
        """
        cal = calibrate_thresholds(collect_points(self.controller, dev_set))
        from .finalizer import Finalizer
        self.controller.tau = cal.tau_answer
        self.controller.tau_abstain = cal.tau_abstain
        self.controller.finalizer = Finalizer(tau_answer=cal.tau_answer,
                                              tau_abstain=cal.tau_abstain)
        return self

    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> Answer:
        """Answer a question with verification, correction, and abstention."""
        state = self.controller.run(question)
        diagnoses = [e["diagnosis"] for e in state.trace if e["event"] == "verify"]
        return Answer(
            question=question,
            answer=state.final_answer,
            status=state.answer_status,
            message=self.controller.finalizer.message(state),
            citations=list(state.citations),
            corrections=state.correction_count,
            diagnoses=diagnoses,
            abstained=state.abstained,
            state=state,
        )

    __call__ = ask
