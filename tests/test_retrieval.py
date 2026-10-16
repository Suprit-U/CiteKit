import pytest

from citekit.chunking import chunk_text
from citekit.embeddings import HashingEmbedder, cosine
from citekit.retrieval import HybridRetriever


@pytest.fixture
def retriever() -> HybridRetriever:
    chunks = chunk_text(
        "\n\n".join(
            [
                "The library is open from 08:00 to 20:00 on weekdays.",
                "Guest wifi login uses the registration number and the portal at wifi.example.edu.",
                "Overdue library books cost Rs 2 per day.",
                "Residents must be inside the block by 23:30 on weekdays.",
            ]
        ),
        size=200,
        overlap=20,
        doc_id="d",
        source="handbook.md",
    )
    return HybridRetriever(chunks)


def test_returns_best_chunk_first(retriever):
    hits = retriever.search("when does the library close", top_k=3)
    assert hits
    assert "20:00" in hits[0].text
    assert hits[0].bm25_rank == 1


def test_top_k_is_respected(retriever):
    assert len(retriever.search("library", top_k=2)) <= 2


def test_scores_are_sorted_descending(retriever):
    scores = [r.score for r in retriever.search("wifi login", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("!!! ???") == []


def test_off_topic_query_is_empty_or_weak(retriever):
    hits = retriever.search("quantum chromodynamics", top_k=3)
    assert all(h.score < 0.05 for h in hits)


def test_dense_only_mode_still_ranks(retriever):
    lexical = retriever.search("wifi portal", top_k=3)
    dense = HybridRetriever(retriever.chunks, use_dense=True).search("wifi portal", top_k=3)
    assert lexical and dense
    assert dense[0].chunk.doc_id == "d"


def test_hashing_embedder_is_deterministic_and_normalised():
    a = HashingEmbedder(dim=128).embed(["wifi portal login"])
    b = HashingEmbedder(dim=128).embed(["wifi portal login"])
    assert a == b
    assert abs(cosine(a[0], a[0]) - 1.0) < 1e-9


def test_cosine_rejects_width_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])