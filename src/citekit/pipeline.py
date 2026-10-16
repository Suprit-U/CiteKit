"""End-to-end pipeline: load -> chunk -> index -> retrieve -> answer -> guard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import guardrails
from .answering import Answer, Answerer, Citation, ExtractiveAnswerer
from .chunking import Chunk, chunk_text
from .config import Settings
from .loaders import Document, load_directory, load_file
from .retrieval import HybridRetriever, RetrievedChunk


@dataclass
class AnswerResult:
    """Everything a caller (CLI, API, evaluator) needs about one question."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    support: float = 0.0
    allowed: bool = False
    reason: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    provider: str = "extractive"

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "allowed": self.allowed,
            "support": self.support,
            "reason": self.reason,
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 2),
            "citations": [
                {
                    "marker": c.marker,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                }
                for c in self.citations
            ],
            "retrieved": [
                {"chunk_id": r.id, "score": r.score, "bm25_rank": r.bm25_rank, "dense_rank": r.dense_rank}
                for r in self.chunks
            ],
        }


class CiteKit:
    """A retrieval-augmented question-answering stack over local documents."""

    def __init__(self, settings: Settings | None = None, answerer: Answerer | None = None) -> None:
        self.settings = settings or Settings()
        self.answerer: Answerer = answerer or ExtractiveAnswerer()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._retriever: HybridRetriever | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> "CiteKit":
        docs = load_directory(directory, kwargs.pop("settings", Settings()))
        return cls(**kwargs).index(docs)

    @classmethod
    def from_text(cls, text: str, *, source: str = "inline.md", **kwargs) -> "CiteKit":
        settings = kwargs.get("settings") or Settings()
        doc = Document(id=f"{Path(source).stem}-inline", source=source, text=text)
        return cls(**kwargs).index([doc], settings=settings)

    def index(self, documents: list[Document], settings: Settings | None = None) -> "CiteKit":
        """(Re)build the chunk index and the retriever."""
        if settings is not None:
            self.settings = settings
        self.documents = list(documents)
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(
                chunk_text(
                    doc.text,
                    size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                    doc_id=doc.id,
                    source=doc.source,
                )
            )
        self.chunks = chunks
        self._retriever = (
            HybridRetriever(
                chunks,
                rrf_k=self.settings.rrf_k,
                use_dense=self.settings.use_dense,
                min_bm25_score=self.settings.min_bm25_score,
            )
            if chunks
            else None
        )
        return self

    # ------------------------------------------------------------------ query
    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._retriever is None:
            return []
        return self._retriever.search(question, top_k=top_k or self.settings.top_k)

    def ask(self, question: str, *, top_k: int | None = None, answerer: Answerer | None = None) -> AnswerResult:
        """Answer a question, or refuse with a reason when unsupported."""
        started = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k)
        used = answerer or self.answerer
        result = _run_guardrail(question, chunks, used, self.settings.min_support)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    @property
    def is_indexed(self) -> bool:
        return self._retriever is not None

    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "avg_chunk_chars": round(
                sum(len(c) for c in self.chunks) / len(self.chunks), 1
            )
            if self.chunks
            else 0.0,
            "dense": self.settings.use_dense,
            "provider": getattr(self.answerer, "name", "unknown"),
        }


def _run_guardrail(question: str, chunks: list[RetrievedChunk], answerer: Answerer, min_support: float) -> AnswerResult:
    answer: Answer = answerer.answer(question, chunks)
    decision = guardrails.check(
        answer_text=answer.text,
        support=answer.support,
        threshold=min_support,
        retrieved=len(chunks),
        # An empty marker list means "the answer cited nothing" -- it must reach the
        # gate and block, not degrade into None (which means "not checked").
        cited_markers=list(answer.cited_markers),
        lexical_grounded=any(c.bm25_rank is not None for c in chunks),
        coverage=_coverage(question, chunks),
    )
    return AnswerResult(
        question=question,
        answer=answer.text if decision.allowed else "",
        citations=answer.citations if decision.allowed else [],
        support=decision.support,
        allowed=decision.allowed,
        reason=decision.reason,
        chunks=chunks,
        provider=getattr(answerer, "name", "unknown"),
    )


def _coverage(question: str, chunks: list[RetrievedChunk]) -> float:
    """Share of the question's content words present in lexically grounded chunks."""
    from .bm25 import tokenize

    query_tokens = set(tokenize(question))
    if not query_tokens:
        return 0.0
    grounded = [c for c in chunks if c.bm25_rank is not None]
    if not grounded:
        return 0.0
    corpus_tokens: set[str] = set()
    for chunk in grounded:
        corpus_tokens.update(tokenize(chunk.text))
    return len(query_tokens & corpus_tokens) / len(query_tokens)


def load_corpus(path: str | Path, settings: Settings | None = None) -> list[Document]:
    """Load one file or a whole directory of documents."""
    p = Path(path)
    settings = settings or Settings()
    if p.is_dir():
        return load_directory(p, settings)
    doc = load_file(p, settings)
    return [doc] if doc else []