"""Self-correcting RAG demo (Stages 1-5).

Run from the repo root:

    python examples/run_demo.py

Shows the active backends, then for each question runs the full controller loop:
plan -> retrieve -> rerank -> generate -> verify -> (diagnose -> correct)* ->
finalize. Prints the answer, the diagnosis trail, and the corrections taken.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import (  # noqa: E402
    Controller, Planner, Stage1Pipeline, Verifier,
    calibrate_thresholds, collect_points,
)


def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(here, os.pardir, "data", "sample_corpus.jsonl")
    documents = load_corpus(corpus_path)

    pipe = Stage1Pipeline(planner=Planner())
    pipe.index_documents(documents)

    # Stage 5: calibrate abstention thresholds on a dev set with unanswerables.
    devset_path = os.path.join(here, os.pardir, "data", "dev_set.jsonl")
    dev = load_corpus(devset_path)
    base = Controller(pipe, Verifier(), tau=0.55, max_corrections=3)
    cal = calibrate_thresholds(collect_points(base, dev))
    controller = Controller(pipe, Verifier(), tau=cal.tau_answer,
                            tau_abstain=cal.tau_abstain, max_corrections=3)

    print("Backends:", pipe.backends)
    print("Calibrated:", cal.to_dict())
    print("=" * 72)

    questions = [
        "What nationality is the director of the film Ran?",  # bridge -> correction
        "Who directed Seven Samurai?",                        # clean single-hop
        "What ranking function catches exact keyword matches?",
        "Who won the 2050 World Cup?",                        # unanswerable -> abstain
        "What is the capital of France?",                     # unanswerable here
    ]

    for q in questions:
        state = controller.run(q)
        actions = [e["action"] for e in state.trace if e["event"] == "correct"]
        print(f"\nQ: {q}")
        print(f"  status:      {state.answer_status}")
        print(f"  outcome:     {controller.finalizer.message(state)}")
        print(f"  corrections: {state.correction_count}  {actions}")


if __name__ == "__main__":
    main()
