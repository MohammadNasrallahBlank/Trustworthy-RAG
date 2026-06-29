"""Leakage checks for the verified-unanswerable set (plan v2 §3.4, Task 2)."""

from __future__ import annotations

from srag.evaluation.leakage import answer_leaks, verify_unanswerable


# --------------------------------------------------------------------------- #
# answer_leaks
# --------------------------------------------------------------------------- #
def test_exact_and_normalized_leak_detection():
    chunks = [{"id": "a", "text": "The capital is Paris."}]
    assert answer_leaks("Paris", chunks) is True
    assert answer_leaks("paris.", chunks) is True       # normalized
    assert answer_leaks("Berlin", chunks) is False


def test_normalized_match_is_word_bounded():
    # "art" should not leak just because "Bart" contains the letters.
    chunks = [{"id": "a", "text": "Bart Simpson lives in Springfield."}]
    assert answer_leaks("art", chunks) is False
    assert answer_leaks("Bart", chunks) is True


def test_alias_leak_detection():
    chunks = [{"id": "a", "text": "The film was directed by Akira Kurosawa."}]
    assert answer_leaks("Kurosawa", chunks, aliases=["Akira Kurosawa"]) is True
    assert answer_leaks("Spielberg", chunks, aliases=["Steven Spielberg"]) is False


def test_entailment_spotcheck_can_flag_a_leak():
    # No literal string match, but an entailment model says the evidence
    # supports the answer -> treated as leaked.
    chunks = [{"id": "a", "text": "Lyon is a French city on the Rhone."}]

    class _Ent:
        def entailment(self, premise, hypothesis):
            return 0.9  # always entails

    assert answer_leaks("Paris", chunks, entailment=_Ent()) is True
    assert answer_leaks("Paris", chunks) is False  # no model -> string-only, no leak


# --------------------------------------------------------------------------- #
# verify_unanswerable
# --------------------------------------------------------------------------- #
_DOCS = [
    {"id": "d0", "text": "The Eiffel Tower is a landmark in Paris."},
    {"id": "d1", "text": "Mount Everest is the highest mountain on Earth."},
]


def test_verify_unanswerable_drops_leaked_and_keeps_clean():
    candidates = [
        # answer "Paris" still present in d0 -> should be dropped as leaked.
        {"question": "Where is the Eiffel Tower?", "answers": ["Paris"],
         "answerable": False},
        # answer "Berlin" appears nowhere -> genuinely unanswerable, kept.
        {"question": "What is the capital of Germany?", "answers": ["Berlin"],
         "answerable": False},
    ]
    kept, dropped = verify_unanswerable(_DOCS, candidates)
    kept_qs = {c["question"] for c in kept}
    dropped_qs = {c["question"] for c in dropped}
    assert "What is the capital of Germany?" in kept_qs
    assert "Where is the Eiffel Tower?" in dropped_qs
    assert all("drop_reason" in c for c in dropped)


def test_verify_unanswerable_reads_held_out_answers_key():
    candidates = [
        {"question": "Where is the Eiffel Tower?", "answers": [],
         "held_out_answers": ["Paris"], "answerable": False},
    ]
    kept, dropped = verify_unanswerable(_DOCS, candidates)
    assert len(dropped) == 1 and len(kept) == 0
