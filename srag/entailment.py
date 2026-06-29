"""Entailment / NLI backend for claim-level attribution (doc Check 2).

Given a premise (a retrieved chunk) and a hypothesis (a claim the generator
emitted), return P(premise entails hypothesis) in [0, 1]. This is the core of
the RAGAS/FActScore-style faithfulness check: decompose the answer into atomic
claims and verify each against its cited context.

Primary path: a sentence-transformers CrossEncoder NLI model whose three logits
map to {contradiction, neutral, entailment}. Fallback (offline / no weights): a
lexical entailment heuristic based on content-term containment, which is crude
but monotone enough to drive the cascade and keep tests deterministic.

Per the doc, the entailment checker should be validated against human
faithfulness labels before being wired into control — see
`tests/test_verifier.py::test_entailment_ranks_supported_above_unsupported` for
a minimal sanity check; a real label set plugs in at stage 5.
"""

from __future__ import annotations

import re
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "that", "this",
    "it", "its", "from", "which", "who", "what", "when", "where", "how", "he",
    "she", "they", "his", "her", "their", "also", "had", "has", "have",
}


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


_NEGATIONS = {"not", "no", "never", "none", "cannot", "n't", "without"}


class EntailmentModel:
    """Scores entailment of a hypothesis by a premise."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        prefer_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.backend = "lexical-fallback"
        self._model = None
        self._entail_idx = 1  # set when a real model loads
        if not prefer_fallback:
            self._try_load(model_name)

    def _try_load(self, model_name: str) -> None:
        try:  # pragma: no cover - optional dependency + weights
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            # sentence-transformers NLI label order is
            # ['contradiction', 'entailment', 'neutral'] for this model family.
            labels = getattr(self._model, "config", None)
            id2label = getattr(labels, "id2label", None) if labels else None
            if id2label:
                for i, name in id2label.items():
                    if "entail" in str(name).lower():
                        self._entail_idx = int(i)
            else:
                self._entail_idx = 1
            self.backend = f"nli:{model_name}"
        except Exception:
            self._model = None

    def entailment(self, premise: str, hypothesis: str) -> float:
        """Return P(premise ⊨ hypothesis) in [0, 1]."""
        if self._model is not None:  # pragma: no cover - optional path
            import numpy as np

            logits = self._model.predict([[premise, hypothesis]])
            logits = np.asarray(logits, dtype=float).reshape(-1)
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            return float(probs[self._entail_idx])
        return self._lexical_entailment(premise, hypothesis)

    def entailment_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[float]:
        return [self.entailment(p, h) for p, h in pairs]

    # ------------------------------------------------------------------ #
    # Offline lexical heuristic
    # ------------------------------------------------------------------ #
    def _lexical_entailment(self, premise: str, hypothesis: str) -> float:
        """Fraction of hypothesis content-terms present in the premise.

        Penalized when the two disagree on negation polarity, so a premise that
        negates the hypothesis does not read as support.
        """
        h_terms = _terms(hypothesis)
        if not h_terms:
            return 0.0
        p_terms = _terms(premise)
        coverage = len(h_terms & p_terms) / len(h_terms)

        p_neg = bool(_NEGATIONS & set(_TOKEN.findall(premise.lower())) | _has_contraction_neg(premise))
        h_neg = bool(_NEGATIONS & set(_TOKEN.findall(hypothesis.lower())) | _has_contraction_neg(hypothesis))
        if p_neg != h_neg:
            coverage *= 0.3  # polarity mismatch -> weak/contradictory support
        return float(coverage)


def _has_contraction_neg(text: str) -> set[str]:
    return {"n't"} if "n't" in text.lower() else set()
