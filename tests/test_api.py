"""Tests for the high-level SelfCorrectingRAG API."""
from __future__ import annotations

from srag import SelfCorrectingRAG, Answer

DOCS = [
    "Ran is a 1985 epic film directed by Akira Kurosawa.",
    "Akira Kurosawa was a Japanese filmmaker born in Tokyo in 1910.",
    "Orson Welles was an American director who made Citizen Kane.",
    "The Kintai Bridge in Iwakuni, Japan was built in 1673.",
]


def test_from_documents_accepts_strings_and_answers():
    rag = SelfCorrectingRAG.from_documents(DOCS)
    r = rag.ask("Who directed Ran?")
    assert isinstance(r, Answer)
    assert "Kurosawa" in r.answer or "Kurosawa" in r.message
    assert r.status in ("answered", "hedged")
    assert r.citations


def test_accepts_dict_documents_with_ids():
    docs = [{"id": "a", "text": "Paris is the capital of France.", "source": "geo"}]
    rag = SelfCorrectingRAG.from_documents(docs)
    r = rag.ask("What is the capital of France?")
    assert r.status in ("answered", "hedged")


def test_abstains_when_corpus_cannot_answer():
    rag = SelfCorrectingRAG.from_documents(DOCS)
    r = rag.ask("Who won the 2050 World Cup?")
    assert r.abstained is True and r.status == "abstained"
    assert r.answer == ""


def test_self_corrects_the_bridge_question():
    rag = SelfCorrectingRAG.from_documents(DOCS, mode="self_correcting")
    r = rag.ask("What nationality is the director of the film Ran?")
    # It resolves the bridge and answers the nationality (not just the director).
    assert "Japanese" in (r.answer + r.message)


def test_calibrate_and_trace_helpers():
    rag = SelfCorrectingRAG.from_documents(DOCS).calibrate([
        {"question": "Who directed Ran?", "answerable": True},
        {"question": "Who won the 2050 World Cup?", "answerable": False},
    ])
    r = rag.ask("Who directed Ran?")
    assert "DECISION TRACE" in r.trace_html()
    assert "Final:" in r.trace_markdown()
    assert set(r.to_dict()) >= {"answer", "status", "citations", "diagnoses"}


def test_mode_trustworthy_is_default_and_runs_no_correction_loop():
    rag = SelfCorrectingRAG.from_documents(DOCS)
    assert rag.mode == "trustworthy"
    assert rag.controller.max_corrections == 0
    r = rag.ask("Who directed Ran?")
    assert r.corrections == 0
    assert r.status in ("answered", "hedged", "abstained")


def test_mode_self_correcting_enables_the_loop():
    rag = SelfCorrectingRAG.from_documents(DOCS, mode="self_correcting",
                                           max_corrections=2)
    assert rag.mode == "self_correcting"
    assert rag.controller.max_corrections == 2


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        SelfCorrectingRAG.from_documents(DOCS, mode="turbo")


def test_trace_json_export_schema():
    import json
    rag = SelfCorrectingRAG.from_documents(DOCS, mode="self_correcting")
    tj = rag.ask("What nationality is the director of the film Ran?").trace_json()
    assert set(tj) >= {"question", "answer", "status", "abstained", "citations",
                       "corrections", "diagnoses", "confidence", "evidence", "trace"}
    assert isinstance(tj["trace"], list) and tj["trace"]
    assert all("event" in e for e in tj["trace"])
    json.dumps(tj)  # must be JSON-serializable
