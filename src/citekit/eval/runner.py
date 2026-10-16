"""Golden-set evaluation harness.

A RAG demo that only shows good answers proves nothing. This runner measures
retrieval quality and — more importantly — abstention behaviour, which is the
failure mode that actually hurts users.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline import CiteKit
from . import metrics


@dataclass(frozen=True)
class Case:
    """One golden evaluation case."""

    question: str
    answerable: bool = True
    must_contain: tuple[str, ...] = ()
    relevant_sources: tuple[str, ...] = ()
    id: str = ""


@dataclass
class EvalReport:
    """Aggregated metrics plus the per-case rows that produced them."""

    cases: int = 0
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: dict[str, float] = field(default_factory=dict)
    latency_ms_mean: float = 0.0
    rows: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "retrieval": self.retrieval,
            "answer": self.answer,
            "latency_ms_mean": self.latency_ms_mean,
            "rows": self.rows,
        }

    def render(self) -> str:
        lines = [f"cases: {self.cases}", f"mean latency: {self.latency_ms_mean:.1f} ms"]
        lines.append("retrieval:")
        for key, value in self.retrieval.items():
            lines.append(f"  {key:<12} {value:.3f}")
        lines.append("answer:")
        for key, value in self.answer.items():
            lines.append(f"  {key:<12} {value:.3f}")
        return "\n".join(lines)


def load_cases(path: str | Path) -> list[Case]:
    """Load a golden JSON file: a list of question objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[Case] = []
    for i, row in enumerate(data, start=1):
        cases.append(
            Case(
                question=row["question"],
                answerable=bool(row.get("answerable", True)),
                must_contain=tuple(row.get("must_contain", [])),
                relevant_sources=tuple(row.get("relevant_sources", [])),
                id=row.get("id") or f"case-{i}",
            )
        )
    return cases


def run(
    kit: CiteKit,
    cases: list[Case],
    *,
    k: int = 5,
    top_k: int | None = None,
) -> EvalReport:
    """Execute every case and aggregate retrieval + answer quality."""
    report = EvalReport(cases=len(cases))
    rec_scores: dict[int, list[float]] = {1: [], 3: [], 5: []}
    ans_hits: list[float] = []
    abstained_when_unanswerable: list[float] = []
    refused_when_answerable: list[float] = []
    latencies: list[float] = []

    for case in cases:
        ranked = [chunk.id for chunk in kit.retrieve(case.question, top_k=top_k or k)]
        relevant = _relevant_chunks(kit, case)

        for cutoff in rec_scores:
            rec_scores[cutoff].append(metrics.recall_at_k(ranked, relevant, cutoff))

        result = kit.ask(case.question, top_k=top_k)
        latencies.append(result.latency_ms)

        if case.answerable:
            body = result.answer.lower()
            hit = 1.0 if (not case.must_contain or all(s.lower() in body for s in case.must_contain)) else 0.0
            ans_hits.append(hit)
            refused_when_answerable.append(0.0 if result.allowed else 1.0)
        else:
            abstained_when_unanswerable.append(1.0 if not result.allowed else 0.0)

        report.rows.append(
            {
                "id": case.id,
                "question": case.question,
                "answerable": case.answerable,
                "allowed": result.allowed,
                "support": result.support,
                "recall@5": round(metrics.recall_at_k(ranked, relevant, 5), 3),
                "mrr@5": round(metrics.mrr_at_k(ranked, relevant, 5), 3),
                "hit_answer": (
                    1.0
                    if (not case.must_contain or all(s.lower() in result.answer.lower() for s in case.must_contain))
                    else 0.0
                ),
                "reason": result.reason,
                "latency_ms": round(result.latency_ms, 2),
            }
        )

    report.retrieval = {
        f"recall@{c}": metrics.mean(v) for c, v in rec_scores.items()
    } | {"mrr@5": metrics.mean([float(r["mrr@5"]) for r in report.rows])}
    report.answer = {
        "answerable_hit": metrics.mean(ans_hits),
        "abstain_on_unanswerable": metrics.mean(abstained_when_unanswerable),
        "unnecessary_refusal": metrics.mean(refused_when_answerable),
    }
    report.latency_ms_mean = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    return report


def _relevant_chunks(kit: CiteKit, case: Case) -> set[str]:
    """Chunk ids whose source file is named in the case."""
    if not case.relevant_sources:
        return set()
    wanted = {s.lower() for s in case.relevant_sources}
    return {c.id for c in kit.chunks if Path(c.source).name.lower() in wanted}