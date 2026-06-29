"""Tests for the Stage 1 pipeline and its components.

These use the offline fallbacks (TF-IDF dense, lexical reranker, extractive
generator) so they run anywhere without model weights or network access. They
assert the Stage 1 *contract*, not exact wording:
  - hybrid retrieval surfaces the relevant chunk near the top,
  - the reranker keeps the on-topic chunk first,
  - the generator always emits an answer field and claim-level citations,
  - unanswerable questions are flagged rather than hallucinated.
"""

from __future__ import annotations

import json
import os

import pytest

from srag import (
    Chunk,
    CrossEncoderReranker,
    Embedder,
    GroundedGenerator,
    HybridRetriever,
    Stage1Pipeline,
    chunk_corpus,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")


def load_corpus() -> list[dict]:
    with open(CORPUS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def pipeline() -> Stage1Pipeline:
    pipe = Stage1Pipeline(
        retriever=HybridRetriever(Embedder(prefer_fallback=True)),
        reranker=CrossEncoderReranker(prefer_fallback=True),
        generator=GroundedGenerator(),
    )
    pipe.index_documents(load_corpus())
    return pipe


# ---------------------------------------------------------------- chunking
def test_chunking_carries_metadata_and_sections():
    docs = load_corpus()
    chunks = chunk_corpus(docs)
    assert len(chunks) >= len(docs)
    assert all(c.id and c.source for c in chunks)
    # At least some chunks pick up a section heading.
    assert any(c.section for c in chunks)
    # Ids are unique.
    assert len({c.id for c in chunks}) == len(chunks)


# --------------------------------------------------------------- retrieval
def test_hybrid_retrieval_surfaces_relevant_chunk():
    chunks = chunk_corpus(load_corpus())
    r = HybridRetriever(Embedder(prefer_fallback=True)).index(chunks)
    hits = r.retrieve("Who directed the film Ran?", top_k=10)
    assert hits
    joined = " ".join(h.text for h in hits[:5]).lower()
    assert "kurosawa" in joined
    # Scores are populated.
    assert hits[0].fusion_score > 0


def test_bm25_catches_exact_keyword():
    chunks = chunk_corpus(load_corpus())
    r = HybridRetriever(Embedder(prefer_fallback=True)).index(chunks)
    hits = r.retrieve("Kintai Bridge", top_k=5)
    assert "kintai" in hits[0].text.lower()


# ---------------------------------------------------------------- reranker
def test_reranker_orders_and_reports():
    chunks = chunk_corpus(load_corpus())
    r = HybridRetriever(Embedder(prefer_fallback=True)).index(chunks)
    cand = r.retrieve("Seven Samurai director", top_k=10)
    rr = CrossEncoderReranker(prefer_fallback=True)
    top = rr.rerank("Who directed Seven Samurai?", cand, top_n=3)
    assert len(top) <= 3
    assert "samurai" in top[0].text.lower()
    report = rr.score_report(top)
    assert report["n"] == len(top)
    assert report["top"] >= report["mean"] - 1e-6


# --------------------------------------------------------------- generator
def test_generator_emits_extract_first_schema():
    gen = GroundedGenerator()
    ev = [Chunk(id="x1", text="Akira Kurosawa was a Japanese filmmaker born in Tokyo.")]
    res = gen.generate("What nationality was Akira Kurosawa?", ev)
    d = res.to_dict()
    # The contract: answer key always present, claims carry citations.
    assert set(d) == {"answer", "claims", "unsupported_facts", "answerable"}
    assert res.answerable is True
    assert res.answer  # non-empty
    assert all(c.citations for c in res.claims)
    assert all(cid == "x1" for c in res.claims for cid in c.citations)


def test_generator_flags_no_evidence():
    gen = GroundedGenerator()
    res = gen.generate("Anything?", [])
    assert res.answerable is False
    assert res.answer == ""
    assert res.unsupported_facts


def test_generator_coerces_llm_json_and_drops_bad_citations():
    captured = {}

    def fake_llm(question, evidence):
        captured["called"] = True
        return {
            "answer": "Japanese",
            "claims": [
                {"text": "Kurosawa was Japanese", "citations": ["good"]},
                {"text": "hallucinated", "citations": ["does-not-exist"]},
            ],
            "unsupported_facts": [],
            "answerable": True,
        }

    gen = GroundedGenerator(llm=fake_llm)
    ev = [Chunk(id="good", text="Kurosawa was Japanese.")]
    res = gen.generate("Nationality?", ev)
    assert captured["called"]
    assert res.answer == "Japanese"
    # Invalid citation id is stripped; valid one kept.
    cites = [cid for c in res.claims for cid in c.citations]
    assert "good" in cites and "does-not-exist" not in cites


# ------------------------------------------------------------- end-to-end
def test_pipeline_answers_grounded_question(pipeline):
    state = pipeline.run("Who directed Seven Samurai?")
    assert state.answerable is True
    assert "kurosawa" in (state.answer + state.explanation).lower()
    assert state.citations  # at least one chunk cited
    # Every cited id actually exists in the reranked evidence.
    ev_ids = {c.id for c in state.evidence}
    assert all(cid in ev_ids for cid in state.citations)


def test_pipeline_flags_unanswerable(pipeline):
    state = pipeline.run("What is the capital of France?")
    assert state.answerable is False


def test_pipeline_trace_records_each_stage(pipeline):
    state = pipeline.run("What nationality is the director of Ran?")
    events = [e["event"] for e in state.trace]
    assert events == ["plan", "retrieve", "rerank", "generate"]


def test_seen_chunk_ids_has_no_duplicates(pipeline):
    state = pipeline.run("Who directed Ran?")
    ids = [c.id for c in state.evidence]
    assert len(ids) == len(set(ids))
