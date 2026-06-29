"""Hybrid retriever: BM25 (sparse) + dense embeddings, fused with RRF.

Doc section 4.3. Sparse catches exact entity/keyword matches; dense catches
paraphrase/semantic matches; Reciprocal Rank Fusion combines the two ranked
lists without needing the two score scales to be comparable. We retrieve a wide
candidate set per sub-query (recall matters more than precision here) and leave
precision to the reranker.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .state import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
    """Minimal BM25 Okapi over a fixed corpus (no external dependency).

    If `rank_bm25` is installed it is used; otherwise this pure-numpy
    implementation provides equivalent Okapi scoring.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(corpus_tokens) else 0.0
        self.n_docs = len(corpus_tokens)
        df: Counter = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.tf = [Counter(doc) for doc in corpus_tokens]

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n_docs):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> dict[int, float]:
    """Fuse several ranked lists of doc indices into one score map (RRF).

    score(d) = sum over lists of 1 / (k + rank_in_list(d)).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """Index a corpus of chunks and retrieve a fused candidate set per query."""

    def __init__(self, embedder: Embedder | None = None, rrf_k: int = 60):
        self.embedder = embedder or Embedder()
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._bm25: _BM25 | None = None
        self._doc_vecs: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> "HybridRetriever":
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        corpus_tokens = [_tokenize(t) for t in texts]
        self._bm25 = _BM25(corpus_tokens)
        self.embedder.fit(texts)
        self._doc_vecs = self.embedder.encode(texts)
        return self

    def retrieve(self, query: str, top_k: int = 50, fusion: str = "rrf") -> list[Chunk]:
        """Return up to `top_k` chunks from BM25 + dense rankings.

        `fusion` selects the combination mode:
          "rrf"    -> Reciprocal Rank Fusion of both rankings (default, hybrid),
          "dense"  -> dense bi-encoder ranking only (the naive baseline),
          "sparse" -> BM25 ranking only.
        Each returned Chunk is a copy carrying its sparse/dense/fusion scores.
        """
        if self._bm25 is None or self._doc_vecs is None:
            raise RuntimeError("HybridRetriever.retrieve called before index().")
        if not self.chunks:
            return []

        # Sparse ranking.
        sparse_scores = self._bm25.scores(_tokenize(query))
        sparse_order = list(np.argsort(-sparse_scores))

        # Dense ranking.
        q_vec = self.embedder.encode([query])[0]
        dense_scores = self.embedder.cosine(q_vec, self._doc_vecs)
        dense_order = list(np.argsort(-dense_scores))

        depth = min(len(self.chunks), max(top_k * 2, 50))
        if fusion == "dense":
            ranked = [(i, float(dense_scores[i])) for i in dense_order[:top_k]]
        elif fusion == "sparse":
            ranked = [(i, float(sparse_scores[i])) for i in sparse_order[:top_k]]
        else:
            # Fuse the two rankings (truncate each to a sane depth before fusing).
            fused = reciprocal_rank_fusion(
                [sparse_order[:depth], dense_order[:depth]], k=self.rrf_k
            )
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

        results: list[Chunk] = []
        for doc_idx, fusion_score in ranked:
            src = self.chunks[doc_idx]
            c = Chunk(
                id=src.id,
                text=src.text,
                source=src.source,
                section=src.section,
                timestamp=src.timestamp,
                sparse_score=float(sparse_scores[doc_idx]),
                dense_score=float(dense_scores[doc_idx]),
                fusion_score=float(fusion_score),
            )
            results.append(c)
        return results
