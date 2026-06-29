"""Cross-encoder reranker (doc section 4.4) — "the cheap win".

A cross-encoder scores each (query, chunk) pair jointly and reorders the
candidate set, keeping the top-N for generation. This is the single
highest-leverage component, and the reason any correction loop must justify
itself *on top of* a reranked baseline.

It also yields the first verification signal for free: the score distribution
of the top-N (absolute level + margin), exposed via `score_report` so a later
verifier can flag a likely retrieval fault *before* spending a generation call.

Primary path: a sentence-transformers CrossEncoder. Fallback (offline / no
weights): a token-overlap lexical scorer. The active backend is reported via
`.backend`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .state import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it",
    "from", "which", "who", "what", "when", "where",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        prefer_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.backend = "lexical-fallback"
        self._model = None
        if not prefer_fallback:
            self._try_load(model_name)

    def _try_load(self, model_name: str) -> None:
        try:  # pragma: no cover - optional dependency + weights
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            self.backend = f"cross-encoder:{model_name}"
        except Exception:
            self._model = None

    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int = 6) -> list[Chunk]:
        """Score and reorder `chunks`; return the top_n with `rerank_score` set."""
        if not chunks:
            return []
        scores = self._score(query, [c.text for c in chunks])
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda cs: -cs[1])
        out: list[Chunk] = []
        for chunk, score in scored[:top_n]:
            chunk.rerank_score = float(score)
            out.append(chunk)
        return out

    def _score(self, query: str, texts: Sequence[str]) -> list[float]:
        if self._model is not None:  # pragma: no cover - optional path
            pairs = [[query, t] for t in texts]
            raw = self._model.predict(pairs)
            return [float(x) for x in raw]
        return self._lexical_scores(query, texts)

    def _lexical_scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Deterministic offline scorer: IDF-weighted token overlap, in [0,1].

        Not a real cross-encoder, but monotone in relevance for keyword-y
        queries, which keeps the pipeline meaningful when weights are absent.
        """
        q_tokens = _tokens(query)
        if not q_tokens:
            return [0.0] * len(texts)
        doc_token_sets = [set(_tokens(t)) for t in texts]
        n = len(texts)
        df: Counter = Counter()
        for s in doc_token_sets:
            for tok in s:
                df[tok] += 1
        idf = {tok: math.log((1 + n) / (1 + df.get(tok, 0))) + 1.0 for tok in set(q_tokens)}
        max_possible = sum(idf.values()) or 1.0
        scores: list[float] = []
        for doc_tokens in doc_token_sets:
            s = sum(idf[tok] for tok in set(q_tokens) if tok in doc_tokens)
            scores.append(s / max_possible)
        return scores

    @staticmethod
    def score_report(reranked: Sequence[Chunk]) -> dict:
        """Cheap pre-generation retrieval-quality signal (doc Check 1).

        Returns the top score, the mean, and the margin between the best and the
        second-best — low absolute scores or a flat margin indicate a likely
        retrieval fault.
        """
        if not reranked:
            return {"top": 0.0, "mean": 0.0, "margin": 0.0, "n": 0}
        scores = [c.rerank_score for c in reranked]
        top = scores[0]
        margin = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]
        return {
            "top": float(top),
            "mean": float(sum(scores) / len(scores)),
            "margin": float(margin),
            "n": len(scores),
        }
