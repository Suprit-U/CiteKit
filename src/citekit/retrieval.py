"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

Why RRF instead of score blending: BM25 scores and cosine similarities are not
on comparable scales, and their distributions shift with corpus size. RRF only
uses *ranks*, so it stays stable without per-corpus tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .embeddings import Embedder, HashingEmbedder, cosine


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk that survived retrieval, with its fused score and signals."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    """Rank chunks for a query using fused lexical and dense signals."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        use_dense: bool = True,
        min_bm25_score: float = 0.5,
    ) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever with no chunks")
        self.chunks = list(chunks)
        self.rrf_k = rrf_k
        self.use_dense = use_dense
        self.min_bm25_score = min_bm25_score

        self.tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Index(self.tokenized)
        self.embedder: Embedder | None = None
        self.vectors: list[list[float]] = []
        if use_dense:
            self.embedder = embedder or HashingEmbedder()
            self.vectors = self.embedder.embed([c.text for c in self.chunks])

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks ranked by fused score."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        lexical = self.bm25.rank(query_tokens)
        lexical = [pair for pair in lexical if pair[1] >= self.min_bm25_score]
        bm25_rank = {idx: rank for rank, (idx, _) in enumerate(lexical, start=1)}
        bm25_scores = dict(lexical)

        dense_rank: dict[int, int] = {}
        if self.use_dense and self.embedder is not None and self.vectors:
            query_vec = self.embedder.embed([query])[0]
            scored = [
                (i, cosine(query_vec, vec)) for i, vec in enumerate(self.vectors)
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            dense_rank = {
                idx: rank
                for rank, (idx, similarity) in enumerate(scored, start=1)
                if _positive(similarity)
            }

        fused: dict[int, float] = {}
        for idx in set(bm25_rank) | set(dense_rank):
            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[idx])
            if idx in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[idx])
            fused[idx] = score

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                score=round(score, 6),
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
                bm25_score=round(bm25_scores.get(idx, 0.0), 4),
            )
            for idx, score in ordered
        ]


def _positive(value: float) -> bool:
    return value > 0.05