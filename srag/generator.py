"""Grounded generation with claim-level citations (doc section 4.5).

The generator's contract is strict, because output discipline is what prevents
Failure 3 (truncation):

  * Cite at the claim level -- each factual claim references the chunk id(s)
    that support it.
  * Extract-then-answer -- the minimal answer span lives in a guaranteed-present
    field, emitted *before* any long explanation, so it survives truncation.
  * Answer only from evidence -- a missing fact is declared, not filled from
    parametric memory (this becomes a clean coverage signal downstream).

`GroundedGenerator` is pluggable: pass an `llm` callable to use a real local
model that returns the JSON schema, or rely on the default deterministic
extractive generator, which honors the same contract without any model weights
so the system runs offline. The schema (`GenerationResult`) is identical either
way.

`prompt_mode="abstain"` (plan v2 section 3.1) adds the one-line instruction that
defines the `prompted_abstain_rag` baseline: refuse when the context lacks the
answer. The reply is parsed against a small FROZEN set of abstention phrases.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional, Sequence

from .state import Chunk, Claim, GenerationResult

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it",
    "from", "which", "who", "what", "when", "where", "did", "do", "does",
    "director", "film", "movie", "nationality",
}

# A callable that takes (question, evidence) and returns a schema-shaped dict.
LLMFn = Callable[[str, Sequence[Chunk]], dict]


SYSTEM_PROMPT = """You answer strictly from the provided evidence chunks.
Return ONLY JSON matching this schema:
{
  "answer": "<minimal answer span, always present>",
  "claims": [{"text": "<atomic factual statement>", "citations": ["<chunk_id>"]}],
  "unsupported_facts": ["<fact you could not ground in evidence>"],
  "answerable": <true|false>
}
Rules:
- Put the shortest correct answer span in "answer" FIRST. Never leave it empty
  unless the question is unanswerable from the evidence (then set answerable=false).
- Every claim must cite the chunk id(s) that support it. If a needed fact is not
  in the evidence, list it under "unsupported_facts" and do NOT invent a citation.
- Do not use outside knowledge."""

# The `prompted_abstain_rag` baseline (plan v2 section 3.1): a one-line prompt
# change asking the model to refuse when the context lacks the answer. The
# instruction requests *exactly* "I don't know"; the parser below accepts a
# small, FROZEN controlled set so the baseline is neither strawmanned nor
# secretly tuned.
ABSTAIN_INSTRUCTION = (
    "\n- If the context does not contain the answer, set answerable=false and "
    'reply with exactly: "I don\'t know".'
)

# Frozen up front -- no post-hoc additions after seeing test (plan v2 section 3.1/3.9).
ABSTENTION_PHRASES = (
    "i don't know",
    "i dont know",
    "the context does not contain",
    "insufficient information",
    "cannot answer from the provided context",
)


def is_abstention(answer: str) -> bool:
    """True if `answer` is one of the frozen controlled abstention phrases.

    Case-insensitive; matches a leading/standalone phrase (trailing punctuation
    and elaboration are allowed, e.g. "Insufficient information." or "I don't
    know -- the passages are about France.").
    """
    a = (answer or "").strip().lower().lstrip("\"'").strip()
    return any(a.startswith(p) for p in ABSTENTION_PHRASES)


class GroundedGenerator:
    """Produce a grounded, extract-first answer with claim-level citations."""

    def __init__(self, llm: Optional[LLMFn] = None, min_relevance: float = 0.05,
                 prompt_mode: str = "grounded"):
        self.llm = llm
        self.min_relevance = min_relevance
        # "grounded" (default) | "abstain" (the prompted-abstain baseline).
        self.prompt_mode = prompt_mode

    def build_prompt(self, question: str, evidence: Sequence[Chunk]) -> str:
        """Render the user prompt (handy for wiring a real local LLM)."""
        blocks = []
        for c in evidence:
            head = f"[{c.id}]" + (
                f" ({c.source}" + (f" / {c.section}" if c.section else "") + ")"
                if c.source else ""
            )
            blocks.append(f"{head}\n{c.text}")
        evidence_block = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
        system = SYSTEM_PROMPT + (ABSTAIN_INSTRUCTION if self.prompt_mode == "abstain" else "")
        return f"{system}\n\nEVIDENCE:\n{evidence_block}\n\nQUESTION: {question}\n\nJSON:"

    def generate(self, question: str, evidence: Sequence[Chunk]) -> GenerationResult:
        if self.llm is not None:
            raw = self.llm(question, evidence)
            result = self._coerce(raw, evidence)
        else:
            result = self._extractive(question, evidence)
        if self.prompt_mode == "abstain" and is_abstention(result.answer):
            # The model declined -> a clean unanswerable signal, regardless of
            # whatever `answerable` flag it returned alongside the refusal.
            result.answer = ""
            result.claims = []
            result.answerable = False
        return result

    # ------------------------------------------------------------------ #
    # Deterministic, offline extractive generator
    # ------------------------------------------------------------------ #
    def _extractive(self, question: str, evidence: Sequence[Chunk]) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                answer="",
                claims=[],
                unsupported_facts=["No evidence was retrieved for this question."],
                answerable=False,
                explanation="",
            )

        q_terms = _content_terms(question)
        scored: list[tuple[float, str, str]] = []  # (score, sentence, chunk_id)
        for c in evidence:
            for sent in _sentences(c.text):
                s_terms = _content_terms(sent)
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms)
                if overlap == 0:
                    continue
                score = overlap / (len(q_terms) or 1)
                scored.append((score, sent, c.id))

        scored.sort(key=lambda x: -x[0])

        if not scored or scored[0][0] < self.min_relevance:
            return GenerationResult(
                answer="",
                claims=[],
                unsupported_facts=[
                    "Retrieved evidence does not appear to address the question."
                ],
                answerable=False,
                explanation="",
            )

        claims: list[Claim] = []
        used_chunks: set[str] = set()
        for score, sent, cid in scored:
            if cid in used_chunks:
                continue
            claims.append(Claim(text=sent.strip(), citations=[cid]))
            used_chunks.add(cid)
            if len(claims) >= 3:
                break

        top_sents = [s for _, s, _ in scored[:5]]
        answer_span = _extract_answer_span(question, scored[0][1], top_sents)
        explanation = " ".join(c.text for c in claims)

        return GenerationResult(
            answer=answer_span,
            claims=claims,
            unsupported_facts=[],
            answerable=True,
            explanation=explanation,
        )

    # ------------------------------------------------------------------ #
    # Coerce an LLM's JSON into the typed schema, defensively.
    # ------------------------------------------------------------------ #
    def _coerce(self, raw, evidence: Sequence[Chunk]) -> GenerationResult:
        if isinstance(raw, str):
            raw = _loads_lenient(raw)
        if not isinstance(raw, dict):
            raw = {}
        valid_ids = {c.id for c in evidence}
        claims = []
        for c in raw.get("claims", []) or []:
            if not isinstance(c, dict):
                continue
            cites = [cid for cid in (c.get("citations") or []) if cid in valid_ids]
            claims.append(Claim(text=str(c.get("text", "")).strip(), citations=cites))
        return GenerationResult(
            answer=str(raw.get("answer", "")).strip(),
            claims=claims,
            unsupported_facts=list(raw.get("unsupported_facts", []) or []),
            answerable=bool(raw.get("answerable", bool(raw.get("answer")))),
            explanation=str(raw.get("explanation", "")).strip(),
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _content_terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _extract_answer_span(
    question: str, sentence: str, candidates_sentences: Optional[Sequence[str]] = None
) -> str:
    """Heuristic minimal-span extraction for the offline generator.

    Prefer a capitalized proper-noun span, but drop proper nouns that already
    appear in the question -- the answer is the *new* entity, not one the asker
    named. That exclusion is what lets a filled bridge hop ("What is the
    nationality of Akira Kurosawa?") resolve to the demonym rather than echoing
    the subject back. For nationality questions specifically we scan the top
    supporting sentences for a demonym (e.g. Japanese), because the answer often
    lives in a sentence that never contains the word "nationality". A real LLM
    generator replaces all of this.
    """
    ql = question.lower()
    q_propers = {p.lower() for p in re.findall(r"\b([A-Z][a-z]+)\b", question)}
    q_words = set(re.findall(r"[a-z]+", ql))

    def _filtered_propers(text: str) -> list[str]:
        props = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        props = [p for p in props if p.lower() not in _STOP]
        return [
            p for p in props
            if not any(w in q_propers or w in q_words for w in p.lower().split())
        ]

    # Nationality / origin questions: look for a demonym across the supporting
    # sentences (single capitalized word with a nationality-like suffix).
    if "national" in ql or "nationality" in ql or ql.startswith("where") and "from" in ql:
        for text in (candidates_sentences or [sentence]):
            for tok in re.findall(r"\b([A-Z][a-z]+)\b", text):
                if tok.lower() in q_propers or tok.lower() in _STOP:
                    continue
                if _is_demonym(tok):
                    return tok

    candidates = _filtered_propers(sentence) or re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", sentence
    )
    if candidates:
        if ql.startswith(("who", "which", "what")) and "national" not in ql:
            return max(candidates, key=len)
        if "national" in ql or ql.startswith("what"):
            return candidates[0]
    return sentence.strip()


# Note: bare "-i" (Iraqi, Israeli) is intentionally excluded -- it produces
# false positives like "Samurai". A real LLM generator has no such issue.
_DEMONYM_SUFFIXES = ("ese", "ian", "ish", "ench", "ican", "ean", "ic")


def _is_demonym(token: str) -> bool:
    t = token.lower()
    if len(t) < 4:
        return False
    return t.endswith(_DEMONYM_SUFFIXES)


def _loads_lenient(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}
