# Contributing to CiteKit

Thanks for taking a look. This project is young and the bar for a change is simple:
**it must keep the library honest about what it does not know.**

## Getting set up

```bash
git clone https://github.com/Suprit-U/CiteKit.git
cd CiteKit
make install
make test
```

You need Python 3.10 or newer. There are no runtime dependencies, so a standard
virtualenv is enough. Nothing in the test suite touches the network.

## Where changes are useful

| Area | Examples |
| :--- | :--- |
| Retrieval | better tokenisation, new fusion strategies, reranking, query expansion |
| Grounding | stronger support metrics, better coverage checks, per-document quotas |
| Answering | streaming answers, structured output, better citation placement |
| Evaluation | more metrics, larger goldens, statistical significance reporting |
| Corpus loading | more formats (PDF, DOCX, notebooks), incremental re-indexing |
| Docs | clearer explanations, worked examples, translations |

## Pull requests

1. Open an issue first for anything larger than a bug fix, so the design can be
   discussed before you write the code.
2. Keep one logical change per pull request.
3. Add a test for every behaviour change. If you change a guardrail, add the case that
   it now blocks.
4. Run `make test` and `make eval`. If the evaluation numbers in `README.md` move,
   include the new output in the pull request description.
5. Write the commit message in the imperative mood and explain *why*, not just *what*.

## What will be rejected

- Changes that make the library answer when the corpus does not support the question.
- New runtime dependencies for small gains. The zero-dependency property is a feature.
- Benchmark numbers without a way to reproduce them.
- Refactors that mix with behaviour changes. Split them.

## Reporting bugs

Use the issue templates. A good report includes the Python version, the corpus you
indexed, the exact question, what CiteKit answered, and what you expected instead.

## Code of conduct

Be direct, be kind, and review the code rather than the person. That is the whole
policy.
