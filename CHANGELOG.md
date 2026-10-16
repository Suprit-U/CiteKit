# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

### Added
- Hybrid retrieval combining BM25 and dense vectors with Reciprocal Rank Fusion.
- Extractive answerer that attaches `[n]` citation markers to every sentence, plus an
  optional LLM answerer whose output is verified against the corpus afterwards.
- Five-stage guardrail layer: retrieval presence, lexical grounding, question coverage,
  citation presence and per-sentence support.
- Evaluation harness with recall@k, hit@k, MRR@k, nDCG@k, precision@k, answerable
  hit rate, refusal rate on unanswerable questions and mean latency.
- `citekit` CLI with `index`, `ask`, `eval` and `serve` subcommands.
- Optional FastAPI server exposing `/health`, `/search` and `/ask`.
- Bundled four-document sample corpus and a ten-case golden evaluation set.
