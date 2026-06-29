"""Watch the self-correcting RAG work, end to end, with REAL open models.

    python examples/run_real_models.py            # uses Qwen2.5-3B-Instruct (cached)
    set DEMO_MODEL=Qwen/Qwen2.5-7B-Instruct & python examples/run_real_models.py

Indexes the small built-in corpus, calibrates abstention on the dev set, then
runs a handful of questions so you can see the loop plan, retrieve, generate,
verify, diagnose, correct, and either answer, hedge, or abstain -- with real
sentence-transformers retrieval, a real cross-encoder reranker, a real NLI
verifier, and a real local LLM generator. This is the "it works" demo.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import (  # noqa: E402
    Controller, CrossEncoderReranker, Embedder, EntailmentModel,
    GroundedGenerator, HybridRetriever, Planner, Stage1Pipeline, Verifier,
    TransformersChat, make_llm_generator_fn, calibrate_thresholds, collect_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def load(name):
    with open(os.path.join(HERE, os.pardir, "data", name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    docs = load("sample_corpus.jsonl")
    dev = load("dev_set.jsonl")
    print(f"Loading real models (LLM={MODEL}) -- first run downloads weights ...",
          flush=True)
    chat = TransformersChat(MODEL)
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=False)),
        reranker=CrossEncoderReranker(prefer_fallback=False),
        generator=GroundedGenerator(llm=make_llm_generator_fn(chat)),
        planner=Planner(),
    ).index_documents(docs)
    ver = Verifier(EntailmentModel(prefer_fallback=False))
    base = Controller(pipe, ver, tau=0.55, max_corrections=2)
    cal = calibrate_thresholds(collect_points(base, dev))
    ctrl = Controller(pipe, ver, tau=cal.tau_answer, tau_abstain=cal.tau_abstain,
                      max_corrections=2)
    print("Backends:", pipe.backends, flush=True)
    print("=" * 72, flush=True)

    questions = [
        "What nationality is the director of the film Ran?",   # bridge -> correction
        "Who directed Seven Samurai?",                         # clean single-hop
        "In what year was the Kintai Bridge built?",           # single-hop
        "Who won the 2050 World Cup?",                         # genuinely unanswerable
    ]
    for q in questions:
        s = ctrl.run(q)
        diagnoses = [e["diagnosis"] for e in s.trace if e["event"] == "verify"]
        actions = [e["action"] for e in s.trace if e["event"] == "correct"]
        print(f"\nQ: {q}", flush=True)
        print(f"  status:      {s.answer_status}", flush=True)
        print(f"  outcome:     {ctrl.finalizer.message(s)}", flush=True)
        print(f"  diagnoses:   {diagnoses}", flush=True)
        print(f"  corrections: {s.correction_count}  {actions}", flush=True)


if __name__ == "__main__":
    main()
