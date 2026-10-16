# CiteKit

**Retrieval-augmented generation that cites its sources and admits when it does not know.**

Most RAG demos will happily answer a question their documents never answered, and
they will do it with confident prose and no evidence. CiteKit is built around the
opposite behaviour: every sentence it returns carries a `[n]` marker pointing at the
exact chunk it came from, and when the corpus does not cover the question it
**refuses** instead of guessing.

- **Hybrid retrieval** — BM25 lexical scoring fused with dense vectors using Reciprocal
  Rank Fusion, so keyword matches and paraphrases both survive.
- **Grounded answers** — support is measured per sentence against *its own* cited
  chunk, not against the whole retrieved pool.
- **Abstention as a feature** — a guardrail layer blocks answers that are ungrounded,
  uncited, or built from chunks that barely touch the question.
- **Zero runtime dependencies.** No API key, no vector database, no model download.
  Deterministic output, runs anywhere Python 3.10+ runs.

---

## Why this exists

A retrieval system that answers everything is indistinguishable from one that
hallucinates. The failure is not that the model is weak — it is that nothing in the
pipeline is allowed to say *no*. CiteKit puts the abstention decision in front of the
user, and it makes that decision measurable with an evaluation harness that ships in
the box.

## Quickstart

```bash
git clone https://github.com/Suprit-U/CiteKit.git
cd CiteKit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

citekit index data/docs
citekit ask "When does the library close on a Friday?"
```

```text
Books on reserve must be collected within three days of the hold expiring [1].
The library is closed on public holidays [1].

cites:   data/docs/library_rules.md#chunk-0 (support 1.00)
support: 1.00   latency: 1.1 ms   grounded: yes
```

Now ask something the corpus does not contain:

```bash
citekit ask "Which bus route goes to the airport on Sunday morning?"
```

```text
refused to answer: retrieved chunks cover only 17% of the question's key terms
$ echo $?
1
```

The non-zero exit code is the point. In a pipeline, refusal is a signal you can route
on; in a chat window it is at least an honest answer.

## Measured results

Real output from `citekit eval evals/golden.json` against the bundled 4-document corpus
(10 chunks), 10 cases — 7 answerable, 3 deliberately unanswerable:

| Metric | Value |
| :--- | ---: |
| recall@1 | 0.300 |
| recall@3 | 0.633 |
| recall@5 | 0.667 |
| MRR@5 | 0.700 |
| answerable questions answered | 1.000 |
| unanswerable questions refused | 1.000 |
| answerable questions wrongly refused | 0.000 |
| mean end-to-end latency | 1.0 ms |

Reproduce it yourself — the harness is part of the package:

```bash
citekit eval evals/golden.json --json
```

Two numbers here matter more than recall: **1.000 on refusal** and **0.000 on
wrongful refusal**. A system that abstains correctly is worth more than one that is
right 70% of the time and confidently wrong the rest.

## How it works

```
documents ──▶ normalise ──▶ chunk ──▶ ┌─ BM25 index ────┐
                                       └─ dense vectors ─┴─▶ RRF fusion ──▶ guardrails ──▶ answer
```

1. **Load & normalise** — HTML is stripped, whitespace collapsed, oversize and
   unsupported files skipped. Every document gets a stable id (slug + content hash),
   so re-indexing an unchanged folder produces identical ids.
2. **Chunk** — paragraphs are split into sentence units and packed to a character
   budget with overlap. Overlap is trimmed against the current chunk, and a single
   sentence longer than the budget is hard-split, so every chunk stays within `size`.
3. **Retrieve** — BM25 (k1 1.5, b 0.75) and cosine similarity over hashed
   character-n-gram embeddings are fused with Reciprocal Rank Fusion
   (`1/(k + rank)`), which needs no score normalisation between the two systems.
4. **Guard** — five gates, any of which can block the answer:
   - nothing retrieved
   - **lexical grounding** — at least one chunk must have matched lexically, not only
     by vector similarity (this is what stops a fuzzy embedding match from carrying a
     question the corpus never mentions)
   - **coverage** — the grounded chunks must cover at least 50% of the question's
     content tokens
   - citations present on every sentence
   - per-sentence support ≥ `min_support`
5. **Answer** — by default extractively, selecting sentences and attaching `[n]`
   markers. An LLM answerer is available for people who have a provider; its output is
   put through the same support check afterwards, so an off-corpus answer is rejected
   even if the model was confident.

## Configuration

```python
from citekit import CiteKit

kit = CiteKit.from_directory("data/docs")   # tuning goes through Settings(...)
result = kit.ask("What is the borrowing limit for a final-year student?")
print(result.answer, result.support, result.citations)
```

| Setting | Default | Meaning |
| :--- | ---: | :--- |
| `chunk_size` | 700 | character budget per chunk |
| `chunk_overlap` | 140 | characters repeated between chunks |
| `top_k` | 5 | chunks considered after fusion |
| `rrf_k` | 60 | Reciprocal Rank Fusion smoothing constant |
| `min_support` | 0.6 | minimum per-sentence support to allow an answer |
| `use_dense` | True | enable the vector side of the hybrid retriever |
| `embedder_dim` | 384 | hashing embedding width (declared on `Settings`; construct `HashingEmbedder(dim=...)` to actually change it) |

## HTTP API

```bash
pip install -e ".[server]"
citekit serve --port 8000 --docs data/docs
```

| Method | Path | Returns |
| :--- | :--- | :--- |
| GET | `/health` | index status, document and chunk counts |
| GET | `/search?q=` | ranked chunks with fused and component scores |
| POST | `/ask` | `{answer, citations, support, refused}` |

## CLI

```text
citekit index <path>              build or rebuild the index
citekit ask <question>            answer with citations, exit 1 if refused
citekit eval <cases.json>         run the evaluation harness (--json for machine output)
citekit serve --port 8000         optional FastAPI server
```

## Tests

```bash
make test     # 48 tests
make eval     # regenerate the numbers in this README
```

The suite covers chunk-budget edge cases, BM25 tokenisation and plural folding, RRF
ordering, each guardrail gate, citation attachment, the eval metrics against hand
computed values, and the CLI exit codes. No network access is required to run it.

## Project layout

```text
src/citekit/
  config.py       frozen Settings dataclass, validated on construction
  loaders.py      HTML stripping, normalisation, stable document ids
  chunking.py     sentence units, overlap packing, position tracking
  bm25.py         tokenizer with stopwords and naive plural folding, BM25 index
  embeddings.py   hashing embedder (default) and optional sentence-transformers
  retrieval.py    hybrid retriever, Reciprocal Rank Fusion
  answering.py    extractive and LLM answerers, support measurement
  guardrails.py   the five gates
  pipeline.py     CiteKit facade
  eval/           recall@k, hit@k, MRR@k, nDCG@k, precision@k and the runner
  cli.py          index / ask / eval / serve
```

## Limitations

Stated plainly, because a tool that hides these is not trustworthy:

- The default `HashingEmbedder` is lexical, not semantic. It catches paraphrases that
  share character n-grams, not true synonymy. Install `sentence-transformers` and pass
  a real embedder when you need semantic recall.
- The default answerer is extractive. It will not write fluent prose, because doing so
  is exactly the step where grounding tends to break.
- The bundled evaluation corpus is small. The numbers above are a regression baseline,
  not a benchmark claim.
- English-language tokenisation only.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).