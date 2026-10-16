"""Embedders behind one tiny interface.

``HashingEmbedder`` is the default: deterministic, offline, no model download,
good enough for the paraphrase cases CiteKit targets. ``SentenceTransformerEmbedder``
is opt-in for users who already run a real encoder.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-width vector."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _ngrams(text: str, lo: int = 3, hi: int = 5) -> list[str]:
    padded = f" {text} "
    return [padded[i : i + n] for n in range(lo, hi + 1) for i in range(max(1, len(padded) - n + 1))]


class HashingEmbedder:
    """Signed random projection over character n-grams (the hashing trick)."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 16:
            raise ValueError("dim too small for a stable embedding space")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in _ngrams(text.lower().strip()):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = struct.unpack("<Q", digest)[0]
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


class SentenceTransformerEmbedder:
    """Optional wrapper around ``sentence-transformers``.

    Imported lazily so the core package keeps zero dependencies.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "sentence-transformers is not installed; `uv add sentence-transformers`"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-width vectors."""
    if len(a) != len(b):
        raise ValueError("vector width mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)