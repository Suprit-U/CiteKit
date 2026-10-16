"""Corpus loading and sentence/word utilities."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t]+")

DEFAULT_SETTINGS = Settings()


@dataclass(frozen=True)
class Document:
    """A single loaded source document."""

    id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def strip_html(raw: str) -> str:
    """Return readable text from an HTML fragment."""
    without_scripts = _SCRIPT_RE.sub(" ", raw)
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", without_scripts))).strip()


def normalize_text(raw: str, *, source: str = "") -> str:
    """Collapse whitespace while preserving paragraph breaks."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if source.lower().endswith((".html", ".htm")):
        text = strip_html(text)
    lines = [_WS_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out).strip()


def document_id(source: str, text: str) -> str:
    """Stable id: source path plus a content digest."""
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", Path(source).stem).strip("-").lower() or "doc"
    return f"{slug}-{digest}"


def load_file(path: str | Path, settings: Settings = DEFAULT_SETTINGS) -> Document | None:
    """Load one file, or return ``None`` if it is skipped."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size > settings.max_doc_bytes or p.suffix.lower() not in settings.extensions:
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = normalize_text(raw, source=str(p))
    if not text:
        return None
    return Document(id=document_id(str(p), text), source=str(p), text=text)


def load_directory(
    directory: str | Path, settings: Settings = DEFAULT_SETTINGS, *, recursive: bool = True
) -> list[Document]:
    """Load every supported text file under ``directory``."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    pattern = "**/*" if recursive else "*"
    docs: list[Document] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.name.startswith("."):
            continue
        doc = load_file(path, settings)
        if doc is not None:
            docs.append(doc)
    return docs