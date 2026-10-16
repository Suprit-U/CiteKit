"""The abstention guardrail.

A retriever will always return *something*. The guardrail is what turns that
into an honest answer: if the generated sentences are not actually supported by
the retrieved chunks, CiteKit refuses instead of publishing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailDecision:
    """Verdict for one answer attempt."""

    allowed: bool
    support: float
    threshold: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "support": self.support,
            "threshold": self.threshold,
            "reason": self.reason,
        }


def check(
    *,
    answer_text: str,
    support: float,
    threshold: float,
    retrieved: int,
    cited_markers: list[int] | None = None,
    lexical_grounded: bool = True,
    coverage: float = 1.0,
    min_coverage: float = 0.5,
) -> GuardrailDecision:
    """Decide whether an answer may be returned to the user.

    Three rules keep the answer honest:

    * ``lexical_grounded`` — dense retrieval may *propose* chunks, but at least
      one proposed chunk must also contain the question's own words.
    * ``coverage`` — those grounded chunks must cover at least ``min_coverage``
      of the question's content words. One incidental match ("semester") is not
      an answer to a question about scholarships.
    * ``support`` — the generated sentences must overlap the corpus.
    """
    if not answer_text.strip():
        return GuardrailDecision(False, support, threshold, "no answer was generated")
    if retrieved == 0:
        return GuardrailDecision(False, support, threshold, "nothing was retrieved for this question")
    if not lexical_grounded:
        return GuardrailDecision(
            False, support, threshold, "no lexical grounding in the retrieved chunks"
        )
    if coverage < min_coverage:
        return GuardrailDecision(
            False,
            support,
            threshold,
            f"retrieved chunks cover only {coverage:.0%} of the question's key terms",
        )
    if cited_markers is not None and not cited_markers:
        return GuardrailDecision(False, support, threshold, "answer carries no citation")
    if support < threshold:
        return GuardrailDecision(
            False,
            support,
            threshold,
            f"support {support:.2f} is below the {threshold:.2f} threshold",
        )
    return GuardrailDecision(True, support, threshold, "answer is supported by the retrieved chunks")