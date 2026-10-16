"""Sentence splitting and chunking.

Chunks are built from whole paragraphs where possible and fall back to
sentence boundaries, because a chunk that ends mid-sentence produces quotes
that nobody can verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARA_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    id: str
    doc_id: str
    source: str
    text: str
    position: int

    def __len__(self) -> int:
        return len(self.text)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping abbreviations intact enough for QA."""
    flat = " ".join(text.split())
    if not flat:
        return []
    parts = _SENT_RE.split(flat)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str, *, size: int = 700, overlap: int = 140, doc_id: str = "doc", source: str = ""
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of roughly ``size`` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            units.append(para)
            continue
        sentences = split_sentences(para)
        buffer = ""
        for sentence in sentences:
            if len(sentence) > size:
                # A single sentence longer than the budget (no terminal punctuation,
                # or a machine-generated blob) must be hard-split, otherwise the unit
                # -- and every chunk containing it -- would exceed ``size``.
                if buffer:
                    units.append(buffer)
                    buffer = ""
                for offset in range(0, len(sentence), size):
                    units.append(sentence[offset : offset + size])
                continue
            if buffer and len(buffer) + 1 + len(sentence) > size:
                units.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            units.append(buffer)

    chunks: list[Chunk] = []
    buffer = ""
    start = 0
    for unit in units:
        if buffer and len(buffer) + 2 + len(unit) > size:
            chunks.append(_make_chunk(buffer, doc_id, source, start))
            room = max(0, size - len(unit) - 1)
            tail = buffer[-overlap : -overlap + room] if (overlap and room) else ""
            start = max(0, len(buffer) - overlap)
            buffer = f"{tail} {unit}".strip() if tail else unit
        else:
            buffer = f"{buffer} {unit}".strip() if buffer else unit
    if buffer:
        chunks.append(_make_chunk(buffer, doc_id, source, start))
    return chunks


def _make_chunk(text: str, doc_id: str, source: str, position: int) -> Chunk:
    cid = f"{doc_id}#{position}"
    return Chunk(id=cid, doc_id=doc_id, source=source, text=text.strip(), position=position)