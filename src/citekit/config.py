"""Tunable knobs for indexing, retrieval and abstention.

Defaults are deliberately conservative: CiteKit would rather refuse to answer
than return an unsupported sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Configuration for :class:`citekit.pipeline.CiteKit`.

    Attributes:
        chunk_size: target characters per chunk.
        chunk_overlap: characters repeated between neighbouring chunks.
        top_k: chunks handed to the answerer.
        rrf_k: rank-fusion constant; larger values flatten score differences.
        min_bm25_score: absolute BM25 score below which a chunk is ignored.
        min_support: minimum fraction of answer sentences carrying a citation
            before the answerer is allowed to answer.
        embedder_dim: width of the hashing embedder.
        use_dense: enable the dense half of hybrid retrieval.
        max_doc_bytes: files larger than this are skipped during loading.
    """

    chunk_size: int = 700
    chunk_overlap: int = 140
    top_k: int = 5
    rrf_k: int = 60
    min_bm25_score: float = 0.5
    min_support: float = 0.6
    embedder_dim: int = 384
    use_dense: bool = True
    max_doc_bytes: int = 2_000_000
    extensions: tuple[str, ...] = (".md", ".txt", ".markdown", ".html", ".htm")
    stopwords: frozenset[str] = field(default_factory=lambda: frozenset())

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 0.0 <= self.min_support <= 1.0:
            raise ValueError("min_support must be within [0, 1]")