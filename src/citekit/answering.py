"""Answer construction and the abstention guardrail.

Two answerers ship in the box:

* ``ExtractiveAnswerer`` (default) — selects the corpus sentences that best
  answer the question and stitches them together with citation markers. It needs
  no API key, so the whole pipeline is testable offline.
* ``LLMAnswerer`` — renders a strict prompt for any callable that follows the
  ``LLMProvider`` protocol. Support is verified *after* generation, so a model
  that drifts off-corpus gets its answer rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .bm25 import tokenize
from .chunking import split_sentences
from .retrieval import RetrievedChunk

CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    """A pointer from an answer back into the corpus."""

    marker: int
    chunk_id: str
    source: str
    text: str


@dataclass
class Answer:
    """Model-free answer text plus the citations it used."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    support: float = 0.0
    provider: str = "extractive"

    @property
    def cited_markers(self) -> list[int]:
        return [int(m) for m in CITE_RE.findall(self.text)]


@runtime_checkable
class Answerer(Protocol):
    name: str

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal contract: text in, text out."""

    name: str

    def generate(self, prompt: str) -> str: ...


class ExtractiveAnswerer:
    """Pick the best-supported sentences and cite them."""

    name = "extractive"

    def __init__(self, *, max_sentences: int = 3, max_chars: int = 420) -> None:
        self.max_sentences = max_sentences
        self.max_chars = max_chars

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        query_tokens = set(tokenize(question))
        if not query_tokens:
            return Answer(text="", support=0.0, provider=self.name)

        scored: list[tuple[float, int, str]] = []
        for marker, chunk in enumerate(chunks, start=1):
            for sentence in split_sentences(chunk.text):
                tokens = set(tokenize(sentence))
                if not tokens:
                    continue
                overlap = len(tokens & query_tokens) / (len(query_tokens) ** 0.5 or 1)
                scored.append((overlap, marker, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))

        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for overlap, marker, sentence in scored:
            if overlap <= 0.0:
                break
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append((marker, sentence))
            if len(chosen) >= self.max_sentences:
                break

        if not chosen:
            return Answer(text="", support=0.0, provider=self.name)

        body = " ".join(f"{sentence} [{marker}]" for marker, sentence in chosen)
        if len(body) > self.max_chars:
            body = body[: self.max_chars].rsplit(" [", 1)[0] + f" [{chosen[-1][0]}]"

        citations = [
            Citation(marker=i + 1, chunk_id=c.chunk.id, source=c.chunk.source, text=c.text)
            for i, c in enumerate(chunks)
        ]
        support = _support_for_chosen(chosen, chunks)
        return Answer(text=body, citations=citations, support=support, provider=self.name)


class LLMAnswerer:
    """Prompt an LLM with the retrieved context and verify what comes back."""

    name = "llm"
    TEMPLATE = (
        "Answer the question using only the context below.\n"
        "Rules: cite every sentence with the matching [n] marker; "
        "if the context does not contain the answer, reply exactly: NOT FOUND.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    def __init__(self, provider: LLMProvider, *, max_chars: int = 420) -> None:
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must expose generate(prompt) -> str and name")
        self.provider = provider
        self.max_chars = max_chars

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"llm:{self.provider.name}"

    def render_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
        return self.TEMPLATE.format(context=context, question=question)

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        raw = self.provider.generate(self.render_prompt(question, chunks)).strip()
        if raw.upper().startswith("NOT FOUND"):
            return Answer(text="", support=0.0, provider=self.name)
        text = raw[: self.max_chars]
        citations = [
            Citation(marker=i + 1, chunk_id=c.chunk.id, source=c.chunk.source, text=c.text)
            for i, c in enumerate(chunks)
        ]
        support = _support_ratio(text, [c.text for c in chunks])
        return Answer(text=text, citations=citations, support=support, provider=self.name)


def _support_for_chosen(chosen: list[tuple[int, str]], chunks: list[RetrievedChunk]) -> float:
    """Support of each chosen sentence measured against the chunk it cites.

    Measuring against the whole retrieved pool flatters the answer: an
    irrelevant sentence is still "supported" by some other chunk. Scoring each
    sentence against its own citation is the honest version.
    """
    if not chosen:
        return 0.0
    sources = {index: chunk.text for index, chunk in enumerate(chunks, start=1)}
    scores: list[float] = []
    for marker, sentence in chosen:
        haystack = set(tokenize(sources.get(marker, "")))
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        scores.append(len(tokens & haystack) / len(tokens))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _support_ratio(text: str, sources: list[str]) -> float:
    """Fraction of answer sentences that overlap a retrieved chunk."""
    sentences = [s for s in split_sentences(text) if s.strip()]
    if not sentences:
        return 0.0
    haystacks = [set(tokenize(s)) for s in sources]
    supported = 0
    for sentence in sentences:
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        best = max((len(tokens & h) / len(tokens) for h in haystacks if h), default=0.0)
        if best >= 0.6:
            supported += 1
    return round(supported / len(sentences), 3)