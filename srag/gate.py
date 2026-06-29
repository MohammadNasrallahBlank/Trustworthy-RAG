"""Active-retrieval gate (doc section 4.1) — Stage 6b, optional.

Not every query needs retrieval, and retrieving when the model already knows the
answer can *introduce* errors via noisy passages (the Astute-RAG / knowledge-
conflict problem). This is an efficiency optimization, not a correctness
requirement, so it is opt-in (`Stage1Pipeline(gate=RetrievalGate())`).

Two mechanisms, per the doc:

  * Heuristic + classifier (`should_retrieve`): a cheap decision over the query.
    Time-sensitive, entity-heavy, or long-tail questions -> retrieve. Stable
    general knowledge / arithmetic / chit-chat -> may answer directly (and still
    verify against retrieval if confidence is borderline).
  * FLARE-style active retrieval (`flare_spans`): during generation, if the
    model's token-level confidence drops on a span that asserts a fact, trigger
    retrieval for *that span*. Pure structure here — it consumes (span,
    confidence) pairs a real LLM produces, and returns the spans worth
    retrieving for. With the offline extractive generator there are no token
    logprobs, so this is exercised via its inputs in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Signals that a question needs fresh / external evidence.
_TIME_SENSITIVE = re.compile(
    r"\b(today|now|current|currently|latest|recent|recently|this year|"
    r"this month|as of|nowadays|upcoming|yesterday|tomorrow)\b", re.IGNORECASE
)
_YEAR = re.compile(r"\b(19|20|21)\d{2}\b")
# Stable / parametric patterns that often do NOT need retrieval.
_ARITHMETIC = re.compile(r"^\s*(what\s+is\s+)?[-+\d\s()*/.^%]+\??\s*$", re.IGNORECASE)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b",
                       re.IGNORECASE)
_PROPER = re.compile(r"\b[A-Z][a-z]+\b")

# A pluggable classifier: question -> True if retrieval is needed.
Classifier = Callable[[str], bool]


@dataclass
class GateDecision:
    retrieve: bool
    reason: str
    signals: dict = field(default_factory=dict)


@dataclass
class RetrievalGate:
    """Decide whether a query needs retrieval.

    Defaults are deliberately conservative: retrieve unless there is a clear
    parametric/stable signal AND no entity/time/long-tail signal. This matches
    the doc's advice to "keep this simple (always retrieve for the target
    domain)" and only skip when it is clearly safe.
    """

    classifier: Optional[Classifier] = None
    min_proper_nouns_for_entity: int = 1

    def should_retrieve(self, question: str) -> GateDecision:
        if self.classifier is not None:
            r = bool(self.classifier(question))
            return GateDecision(r, "classifier", {"classifier": r})

        q = question.strip()
        signals = {
            "time_sensitive": bool(_TIME_SENSITIVE.search(q) or _YEAR.search(q)),
            "entity_heavy": len(_PROPER.findall(q)) >= self.min_proper_nouns_for_entity,
            "arithmetic": bool(_ARITHMETIC.match(q)),
            "greeting": bool(_GREETING.match(q)),
        }
        # Clear parametric/stable cases -> skip retrieval.
        if signals["greeting"]:
            return GateDecision(False, "greeting/chit-chat", signals)
        if signals["arithmetic"]:
            return GateDecision(False, "arithmetic/parametric", signals)
        # Anything time-sensitive or entity-heavy -> retrieve.
        if signals["time_sensitive"]:
            return GateDecision(True, "time-sensitive", signals)
        if signals["entity_heavy"]:
            return GateDecision(True, "entity-heavy", signals)
        # Default: retrieve (safe for the target domain).
        return GateDecision(True, "default-retrieve", signals)

    # ------------------------------------------------------------------ #
    # FLARE-style active retrieval
    # ------------------------------------------------------------------ #
    @staticmethod
    def flare_spans(spans_with_confidence: Sequence[tuple], threshold: float = 0.5,
                    *, only_factual: bool = True) -> list[str]:
        """Return spans whose token-confidence dropped below `threshold`.

        `spans_with_confidence` is a sequence of (span_text, confidence) as a
        real LLM would emit per upcoming sentence/span. A span is selected for
        active retrieval when its confidence is low and (optionally) it looks
        like it asserts a fact (contains a content word / number / proper noun)
        rather than being filler.
        """
        out: list[str] = []
        for span, conf in spans_with_confidence:
            if conf >= threshold:
                continue
            if only_factual and not _asserts_fact(span):
                continue
            out.append(span.strip())
        return out


def _asserts_fact(span: str) -> bool:
    if re.search(r"\d", span):
        return True
    if _PROPER.search(span):
        return True
    # A bare function-word span is not a factual assertion.
    _FILLER = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "of", "and",
        "or", "to", "in", "it", "that", "this", "for", "on", "as", "by", "with",
        "which", "thing", "things", "something", "stuff", "one", "kind", "sort",
        "there", "here", "they", "we", "you", "he", "she", "so", "such", "very",
    }
    content = [w for w in re.findall(r"[a-z]+", span.lower()) if w not in _FILLER]
    return len(content) >= 2
