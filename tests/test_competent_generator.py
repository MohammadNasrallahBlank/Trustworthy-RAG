"""End-to-end proof with a *competent* (synthesizing) generator.

The default offline generator is extractive — it cannot synthesize a yes/no
answer, and because it only emits claims it copied from chunks, faithfulness is
trivially 1.0. Those are limitations of that stand-in generator, NOT of the
architecture. Here we plug in a deterministic oracle LLM (schema-correct, no
weights) to show two things the architecture actually does once the generator
can synthesize:

  A. it produces a synthesized comparison answer ("yes") with real per-claim
     citations, and that answer is correct; and
  B. when the generator fabricates a claim its cited chunk does not entail, the
     verifier's faithfulness check *catches it* (faithfulness < 1.0) and routes a
     generation fault — i.e. faithfulness is a real, non-trivial signal.
"""

from __future__ import annotations

from srag import (
    Controller, CrossEncoderReranker, Embedder, EntailmentModel,
    GroundedGenerator, HybridRetriever, Planner, Stage1Pipeline, Verifier,
)
from srag.evaluation.metrics import exact_match

CORPUS = [
    {"id": "kurosawa", "source": "Akira Kurosawa",
     "text": "Akira Kurosawa was a Japanese film director and screenwriter who directed thirty films."},
    {"id": "ozu", "source": "Yasujiro Ozu",
     "text": "Yasujiro Ozu was a Japanese film director and screenwriter. His film Tokyo Story is acclaimed."},
    {"id": "welles", "source": "Orson Welles",
     "text": "Orson Welles was an American film director famous for Citizen Kane."},
]

# A deterministic oracle that returns the strict JSON schema, like a real LLM
# would — but with controllable behavior so the test is reproducible.
_ANSWERS = {
    "Were Akira Kurosawa and Yasujiro Ozu both Japanese film directors?": {
        "answer": "yes",
        "claims": [
            ("Akira Kurosawa was a Japanese film director", "kurosawa"),
            ("Yasujiro Ozu was a Japanese film director", "ozu"),
        ],
    },
    # A fabrication: a fact that appears in NO chunk, cited to the Ozu chunk.
    "What Olympic medal did Yasujiro Ozu win in 1960?": {
        "answer": "Gold Medal",
        "claims": [
            ("Yasujiro Ozu won a Gold Medal at the 1960 Olympic Games", "ozu"),
        ],
    },
}


def oracle_llm(question, evidence):
    spec = _ANSWERS.get(question)
    if spec is None:
        return {"answer": "", "claims": [], "unsupported_facts": ["unknown"],
                "answerable": False}
    claims = []
    for text, key in spec["claims"]:
        cid = next((c.id for c in evidence if key in c.id), None)
        claims.append({"text": text, "citations": [cid] if cid else []})
    return {"answer": spec["answer"], "claims": claims,
            "unsupported_facts": [], "answerable": True}


def _faithfulness(state, entailment, threshold=0.5):
    claims = [c for c in state.claims if c.citations]
    if not claims:
        return None
    by_id = {c.id: c for c in state.evidence}
    ent = 0
    for cl in claims:
        best = max((entailment.entailment(by_id[cid].text, cl.text)
                    for cid in cl.citations if cid in by_id), default=0.0)
        if best >= threshold:
            ent += 1
    return ent / len(claims)


def _controller():
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=True)),
        reranker=CrossEncoderReranker(prefer_fallback=True),
        generator=GroundedGenerator(llm=oracle_llm),
        planner=Planner(),
    ).index_documents(CORPUS)
    return Controller(pipe, Verifier(EntailmentModel(prefer_fallback=True)),
                      tau=0.5, max_corrections=2)


def test_competent_generator_synthesizes_correct_comparison_answer():
    ctrl = _controller()
    state = ctrl.run("Were Akira Kurosawa and Yasujiro Ozu both Japanese film directors?")
    # A synthesized yes/no answer the extractive generator could not produce.
    assert exact_match(state.final_answer, ["yes"]) == 1.0
    assert state.answer_status == "answered"
    # Both claims are grounded in their cited chunks -> faithful.
    faith = _faithfulness(state, EntailmentModel(prefer_fallback=True))
    assert faith == 1.0
    cited = " ".join(cid for c in state.claims for cid in c.citations)
    assert "kurosawa" in cited and "ozu" in cited


def test_faithfulness_catches_a_fabricated_claim():
    ctrl = _controller()
    state = ctrl.run("What Olympic medal did Yasujiro Ozu win in 1960?")
    # The fabricated claim is NOT entailed by its cited chunk -> faithfulness < 1.
    faith = _faithfulness(state, EntailmentModel(prefer_fallback=True))
    assert faith is None or faith < 1.0
    # The verifier flags it rather than answering confidently.
    v = Verifier(EntailmentModel(prefer_fallback=True))
    result = v.verify(state)
    assert result.unsupported_claims or state.answer_status in ("abstained", "hedged")
