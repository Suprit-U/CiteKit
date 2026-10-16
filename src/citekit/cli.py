"""Command line interface: ``citekit index | ask | eval | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .eval import run as run_eval
from .eval.runner import load_cases
from .pipeline import CiteKit, load_corpus

BANNER = "citekit - verifiable retrieval-augmented Q&A"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="citekit", description=BANNER)
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="load a corpus and print index statistics")
    index.add_argument("path", help="file or directory")
    index.add_argument("--chunk-size", type=int, default=700)
    index.add_argument("--no-dense", action="store_true", help="BM25 only")

    ask = sub.add_parser("ask", help="ask a question against a corpus")
    ask.add_argument("path")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--json", action="store_true", help="machine-readable output")

    evaluate = sub.add_parser("eval", help="run a golden set and report metrics")
    evaluate.add_argument("path")
    evaluate.add_argument("cases", help="path to golden JSON")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the HTTP API (needs the server extra)")
    serve.add_argument("path")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        chunk_size=getattr(args, "chunk_size", 700) or 700,
        use_dense=not getattr(args, "no_dense", False),
    )


def _build(args: argparse.Namespace) -> CiteKit:
    settings = _settings(args)
    return CiteKit(settings=settings).index(load_corpus(args.path, settings))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        from .server import serve  # imported lazily: optional dependency

        return serve(_build(args), host=args.host, port=args.port)

    kit = _build(args)
    if not kit.is_indexed:
        print(f"no usable documents under {args.path}", file=sys.stderr)
        return 2

    if args.command == "index":
        print(json.dumps(kit.stats(), indent=2))
        return 0

    if args.command == "ask":
        result = kit.ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            if result.allowed:
                print(result.answer)
                for c in result.citations:
                    print(f"  [{c.marker}] {Path(c.source).name} ({c.chunk_id})")
                print(f"support={result.support}  latency={result.latency_ms:.1f} ms")
            else:
                print(f"refused: {result.reason}")
        return 0 if result.allowed else 1

    report = run_eval(kit, load_cases(args.cases), k=args.k)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())