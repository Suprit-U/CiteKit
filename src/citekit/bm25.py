"""A small, dependency-free BM25 implementation.

Sparse lexical retrieval is kept in-tree on purpose: lexical matching is what
catches exact identifiers (Wi-Fi codes, port numbers, course codes) that dense
embeddings routinely miss, and shipping it with no dependency makes the library
easy to audit.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#-]*")

ENGLISH_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    is are was were be been being do does did done have has had can could should would will
    what which who whom whose when where why how it its as not no so such very about into
    over under again further once here there all any both each few more most other some only
    """.split()
)


def tokenize(text: str, *, stopwords: frozenset[str] = ENGLISH_STOPWORDS) -> list[str]:
    """Lowercase word tokens, naive plural folding, stopword removal.

    Folding plurals (``books`` -> ``book``, ``weekdays`` -> ``weekday``) is
    deliberately naive: a full stemmer drags in a dependency and mis-stems the
    identifiers this index is meant to match exactly.
    """
    raw = _TOKEN_RE.findall(text.lower())
    folded = _fold(raw)
    keep = [
        f
        for original, f in zip(raw, folded)
        if original not in stopwords and f not in stopwords and len(f) > 1
    ]
    return keep


def _fold(tokens: list[str]) -> list[str]:
    folded: list[str] = []
    for token in tokens:
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        folded.append(token)
    return folded


@dataclass
class BM25Index:
    """Okapi BM25 over an in-memory list of token lists."""

    k1: float = 1.5
    b: float = 0.75

    def __init__(self, documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25Index needs at least one document")
        self.k1, self.b = k1, b
        self.n_docs = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avgdl = sum(self.doc_len) / self.n_docs
        self.tf: list[Counter[str]] = [Counter(d) for d in documents]
        df: Counter[str] = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: list[str], doc_index: int) -> float:
        """BM25 score of one document against ``query`` tokens."""
        tf, length = self.tf[doc_index], self.doc_len[doc_index]
        score = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
        return score

    def rank(self, query: list[str], *, limit: int | None = None) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs sorted by descending score."""
        scored = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit] if limit else scored