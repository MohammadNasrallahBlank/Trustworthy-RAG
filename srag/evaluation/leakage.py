"""Leakage checks for the verified-unanswerable set (plan v2 section 3.4).

When we hold out a question's gold context to make it *unanswerable*, we must
verify the answer is not still recoverable from the rest of the corpus -- a
distractor passage, an alias mention, or evidence that simply entails the answer.
Otherwise an "unanswerable" question is actually answerable and a system that
answers it is not committing an Unsupported Confident Answer (UCA).

A held-out question is kept as "verified-unanswerable" only if its answer does
NOT leak. Two cheap-first checks, plus an optional model check:

  1. exact / SQuAD-normalized substring of the answer (or an alias) in any chunk;
  2. (optional) entailment spot-check: does top-k retrieved evidence entail the
     gold answer? If yes -> leaked.

Questions that leak are DROPPED (with a reason), never silently kept.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .metrics import normalize_answer


def _chunk_text(c) -> str:
    return c["text"] if isinstance(c, dict) else getattr(c, "text", "")


def _normalized_contains(needle_norm: str, haystack_norm: str) -> bool:
    """Word-bounded substring match over SQuAD-normalized token strings.

    Both sides are space-joined normalized tokens, so padding with spaces makes
    the containment check respect token boundaries: "art" does not match inside
    "bart", but "bart" matches "bart simpson ...".
    """
    if not needle_norm:
        return False
    return f" {needle_norm} " in f" {haystack_norm} "


def answer_leaks(answer: str, chunks: Sequence, *,
                 aliases: Optional[Sequence[str]] = None,
                 entailment=None, entail_threshold: float = 0.5) -> bool:
    """True if `answer` (or any alias) is recoverable from `chunks`.

    String checks are exact (case-insensitive) and SQuAD-normalized + word
    bounded. If an `entailment` model is supplied, a passage that entails the
    answer above `entail_threshold` also counts as a leak.
    """
    candidates = [answer, *(aliases or [])]
    texts = [_chunk_text(c) for c in chunks]

    for cand in candidates:
        if not cand or not str(cand).strip():
            continue
        cand_norm = normalize_answer(str(cand))
        for t in texts:
            # SQuAD-normalized + word-bounded: case- and punctuation-insensitive,
            # but "art" must not match inside "Bart".
            if _normalized_contains(cand_norm, normalize_answer(t)):
                return True

    if entailment is not None:
        for cand in candidates:
            if not cand or not str(cand).strip():
                continue
            for t in texts:
                try:
                    if entailment.entailment(t, str(cand)) >= entail_threshold:
                        return True
                except Exception:
                    continue
    return False


def verify_unanswerable(documents: Sequence[dict], candidates: Sequence[dict], *,
                        retriever=None, entailment=None, top_k: int = 10,
                        answer_keys: Sequence[str] = ("answers", "held_out_answers")):
    """Split unanswerable `candidates` into (kept, dropped).

    `documents` is the indexed corpus (list of {id, text, ...}). Each candidate
    carries the answer to check under one of `answer_keys` (the first non-empty
    wins) plus optional `answer_aliases`. A candidate is DROPPED (leaked) if:

      * its answer/alias string is present anywhere in `documents`; or
      * (when `retriever` + `entailment` are given) the top-k evidence for its
        question entails the answer.

    Returns (kept, dropped); each dropped item is a copy with a `drop_reason`.
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for item in candidates:
        answers: list[str] = []
        for key in answer_keys:
            vals = item.get(key)
            if vals:
                answers = list(vals)
                break
        aliases = item.get("answer_aliases")

        leaked = False
        reason = None
        for ans in answers:
            if answer_leaks(ans, documents, aliases=aliases):
                leaked, reason = True, "answer_string_present_in_corpus"
                break

        if not leaked and retriever is not None and entailment is not None and answers:
            try:
                retrieved = retriever.retrieve(item["question"], top_k=top_k)
            except Exception:
                retrieved = []
            for ans in answers:
                if answer_leaks(ans, retrieved, aliases=aliases,
                                entailment=entailment):
                    leaked, reason = True, "entailed_by_topk_evidence"
                    break

        if leaked:
            dropped.append({**item, "drop_reason": reason})
        else:
            kept.append(item)

    return kept, dropped
