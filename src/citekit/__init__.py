"""CiteKit — verifiable retrieval-augmented Q&A.

The design rule this package follows: *an answer is only allowed to exist if the
corpus supports it*. Every claim is traceable to a chunk id, and the pipeline
abstains (rather than guessing) when support is too weak.

Public surface:

    from citekit import CiteKit, Corpus

    kit = CiteKit.from_directory("data/docs")
    result = kit.ask("What is the guest wifi password policy?")
    if result.guardrail.allowed:
        print(result.answer, result.citations)
"""

from .config import Settings
from .loaders import Document, load_directory, load_file
from .pipeline import AnswerResult, CiteKit
from .retrieval import HybridRetriever, RetrievedChunk
from .version import __version__

__all__ = [
    "AnswerResult",
    "CiteKit",
    "Corpus",
    "Document",
    "HybridRetriever",
    "RetrievedChunk",
    "Settings",
    "__version__",
    "load_directory",
    "load_file",
]


class Corpus:
    """Container for a loaded, indexed set of documents."""

    def __init__(self, documents: list["Document"]) -> None:
        self.documents = list(documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Corpus(documents={len(self.documents)})"