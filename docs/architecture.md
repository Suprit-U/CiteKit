# Architecture

## Design goals

1. **Refusal is a first-class outcome.** Every stage can end the request. Nothing in
   the pipeline is allowed to "best effort" its way to an answer.
2. **Grounding is measured, not assumed.** Support is a number with a threshold, not
   a vibe.
3. **No hidden network calls.** The default path is fully offline and deterministic.
   The same corpus and the same question produce the same answer.

## Stages

### 1. Loading (`loaders.py`)

`load_directory` walks a tree, skips dotfiles, unsupported extensions and files above
`max_doc_bytes`, and raises `NotADirectoryError` when the path is not a directory —
silently indexing nothing is worse than failing.

`document_id` is `slug-of-filename-sha1-of-content[:8]`. The content hash means the id
changes when the content changes and stays stable across re-index runs, which is what
makes citations stable.

HTML is stripped with a small tag-removal pass and entity unescaping. This is not a
parser-grade extractor and the README says so; a real deployment should swap in a
proper reader behind the same interface.

### 2. Chunking (`chunking.py`)

Text is reduced to sentence units, then packed greedily into chunks under
`chunk_size` characters. When a unit does not fit, the overlap tail is trimmed to
`room = chunk_size - len(unit) - 1` before being carried forward, which guarantees the
budget is respected even when a single sentence is large.

`Chunk.position` is an offset into the accumulated unit buffer used while packing the
chunk, not an index into the original document -- do not use it to slice the source
file. Each chunk also carries `doc_id` and `source`, which are what a citation shows.
point at a location rather than just a document.

### 3. Lexical retrieval (`bm25.py`)

The tokenizer keeps technical tokens intact (`a-z0-9` plus `_ + . # -`) so
`node.js`, `c++` and `utf-8` survive, applies a stopword list, and folds naive
plurals. The stopword check runs against **both** the raw and the folded token —
without that, `does` folds to `doe` and leaks past the stoplist.

BM25 uses the standard `k1=1.5`, `b=0.75`. `rank()` returns a dense 1-based ordering,
because the fusion step consumes ranks, not scores.

### 4. Dense retrieval (`embeddings.py`, `retrieval.py`)

The default `HashingEmbedder` hashes character 3- to 5-grams with blake2b, applies the
signed hashing trick to keep the inner product unbiased, and L2-normalises. It is
deterministic, dependency-free and good enough for lexical near-duplicates. It is not
a semantic model and the code, the README and the tests all say so.

`SentenceTransformerEmbedder` is the opt-in upgrade.

### 5. Fusion (`retrieval.py`)

Reciprocal Rank Fusion scores a chunk as `1/(rrf_k + rank)` from each retriever and
sums the contributions. RRF was chosen over score normalisation because BM25 scores
and cosine similarities are not on comparable scales, and because rank fusion degrades
gracefully when one retriever has nothing to say about a query.

Dense-only hits are dropped from the fused list, and separately re-checked by the
lexical-grounding gate, so a vector-only fuzzy match can never carry a question the
corpus does not mention.

### 6. Guardrails (`guardrails.py`)

| Gate | Question it answers |
| :--- | :--- |
| `empty` | did we produce any text at all? |
| `retrieved` | did anything come back? |
| `lexical_grounded` | did at least one chunk match on words, not only vectors? |
| `coverage` | do the grounded chunks cover enough of the question's content tokens? |
| `citations` | is every sentence traceable? |
| `support` | does each sentence's overlap with its own cited chunk clear `min_support`? |

`coverage` uses the content tokens of the question — stopwords removed — and requires
50% of them to appear in the grounded chunks. This is the gate that caught the real
failure mode during development: a dense-only match scored 0.9 similarity on a question
about "airport routes" and the corpus was about library hours. High similarity, zero
coverage, refuse.

### 7. Answering (`answering.py`)

The extractive answerer ranks sentences by their overlap with the question, keeps the
best-scoring ones (it does **not** enforce one sentence per chunk), and appends `[n]`. Support for a chosen sentence is
measured against **its own cited chunk** — an earlier version scored every sentence
against the whole retrieved pool, which inflated support for sentences that matched a
different chunk. That was a bug, and the tests now pin the behaviour.

`LLMAnswerer` wraps any provider behind `LLMProvider`. The model is instructed to
answer only from the context and to emit `NOT FOUND` when the context is insufficient.
Its output is still put through the same support check, so a confident off-corpus
answer is rejected after the fact.

## Deliberate non-goals

- **A vector database.** With the corpus sizes this targets, an in-memory index is
  faster, reproducible and has no service to operate.
- **A framework.** No LangChain-style indirection. The stages are plain functions and
  the data flow is readable in one sitting.
- **Fluency at the cost of grounding.** The extractive default is less pleasant to read
  than a generated paragraph. That is the trade being made on purpose.
