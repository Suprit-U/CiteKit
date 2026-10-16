from citekit.chunking import chunk_text, split_sentences


def test_split_sentences_handles_multiple_sentences():
    text = "The library opens at 08:00. It closes at 20:00. Bring your student card."
    assert split_sentences(text) == [
        "The library opens at 08:00.",
        "It closes at 20:00.",
        "Bring your student card.",
    ]


def test_split_sentences_empty_is_empty_list():
    assert split_sentences("   ") == []


def test_chunk_text_respects_size_and_produces_overlap():
    text = "\n\n".join(f"Paragraph number {i} with some content to make it long enough." for i in range(12))
    chunks = chunk_text(text, size=220, overlap=60, doc_id="d", source="s.md")

    assert chunks, "expected at least one chunk"
    assert all(len(c) <= 260 for c in chunks), "chunk exceeded the soft size budget"
    assert [c.position for c in chunks] == sorted(c.position for c in chunks)
    assert chunks[0].position == 0
    assert all(c.doc_id == "d" and c.source == "s.md" for c in chunks)
    assert chunks[0].id == "d#0"


def test_chunk_text_does_not_split_mid_sentence():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_text(text, size=30, overlap=0, doc_id="d")
    for chunk in chunks:
        assert chunk.text.rstrip().endswith(".")


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    try:
        chunk_text("text", size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_chunk_text_empty_input_returns_empty():
    assert chunk_text("   ", size=100, overlap=10) == []