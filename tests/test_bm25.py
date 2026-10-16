import math

from citekit.bm25 import BM25Index, tokenize


def test_tokenize_drops_stopwords_and_lowercases():
    tokens = tokenize("The Guest Network is on campus-guest")
    assert "the" not in tokens
    assert "guest" in tokens
    assert "campus-guest" in tokens


def test_tokenize_keeps_identifiers():
    assert "20:00" not in tokenize("closes at 20:00")  # colon splits the token
    assert "300" in tokenize("up to 300 Mbps")


def test_bm25_ranks_matching_document_first():
    docs = [
        tokenize("library opens at 08:00 and closes at 20:00"),
        tokenize("wifi portal login uses the student registration number"),
        tokenize("overdue fines are two rupees per day"),
    ]
    index = BM25Index(docs)
    ranked = index.rank(tokenize("library closing time"))
    assert ranked and ranked[0][0] == 0
    assert ranked[0][1] > 0


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index([tokenize("library hours"), tokenize("wifi policy")])
    assert index.rank(tokenize("helicopter")) == []


def test_bm25_idf_is_positive_and_finite():
    index = BM25Index([tokenize("alpha beta"), tokenize("beta gamma"), tokenize("delta")])
    for term, idf in index.idf.items():
        assert math.isfinite(idf) and idf > 0, (term, idf)


def test_bm25_requires_documents():
    try:
        BM25Index([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bm25_score_is_zero_for_doc_without_query_terms():
    index = BM25Index([tokenize("alpha beta"), tokenize("gamma delta")])
    assert index.score(tokenize("alpha"), 1) == 0.0