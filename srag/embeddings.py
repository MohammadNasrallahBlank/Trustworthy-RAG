"""Dense embedding backend (doc section 4.3, "dense bi-encoder").

Primary path: sentence-transformers (a real open bi-encoder). If the library
or its model weights are unavailable (e.g. an offline/air-gapped box), this
falls back to a deterministic, dependency-light TF-IDF vectorizer so the rest
of the pipeline still runs and tests pass. The fallback is *not* a substitute
for a real encoder in production — it exists purely to keep Stage 1 runnable
without network access, and the active backend is reported via `.backend`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# A tiny, conventional English stopword list for the fallback vectorizer.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "that", "this",
    "it", "its", "from", "which", "who", "what", "when", "where", "how",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Embedder:
    """Encodes texts into L2-normalized dense vectors.

    Args:
        model_name: sentence-transformers model id to try first.
        prefer_fallback: force the TF-IDF fallback (useful for tests / offline).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        prefer_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.backend = "tfidf-fallback"
        self._model = None
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._fitted = False

        if not prefer_fallback:
            self._try_load_sentence_transformer(model_name)

    def _try_load_sentence_transformer(self, model_name: str) -> None:
        try:  # pragma: no cover - depends on optional dependency + weights
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.backend = f"sentence-transformers:{model_name}"
        except Exception:
            # Library missing or weights can't be downloaded -> fallback.
            self._model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, corpus: Sequence[str]) -> "Embedder":
        """Fit the fallback vectorizer's vocabulary/IDF on a corpus.

        A no-op for the real sentence-transformer backend, so callers can always
        call `fit` then `encode` regardless of which backend is active.
        """
        if self._model is not None:
            self._fitted = True
            return self

        docs_tokens = [_tokenize(t) for t in corpus]
        df: Counter = Counter()
        for toks in docs_tokens:
            for tok in set(toks):
                df[tok] += 1
        self._vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n_docs = max(1, len(docs_tokens))
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for tok, i in self._vocab.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
        self._idf = idf
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        if self._model is not None:  # pragma: no cover - optional path
            vecs = np.asarray(
                self._model.encode(list(texts), normalize_embeddings=True),
                dtype=np.float32,
            )
            return vecs
        return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._idf is None:
            raise RuntimeError("Embedder fallback used before .fit(); call fit(corpus) first.")
        dim = len(self._vocab)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = _tokenize(text)
            if not toks:
                continue
            tf = Counter(toks)
            for tok, count in tf.items():
                j = self._vocab.get(tok)
                if j is None:
                    continue
                out[r, j] = (1.0 + math.log(count)) * self._idf[j]
            norm = np.linalg.norm(out[r])
            if norm > 0:
                out[r] /= norm
        return out

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between a single query vector and a matrix of rows.

        Assumes inputs are already L2-normalized (as produced by `encode`).
        """
        return b @ a
