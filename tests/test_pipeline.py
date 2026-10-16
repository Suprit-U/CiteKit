from citekit.answering import ExtractiveAnswerer, LLMAnswerer, _support_ratio
from citekit.config import Settings
from citekit.guardrails import check


class FakeProvider:
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_support_ratio_is_one_for_verbatim_copy():
    source = ["The library closes at 20:00 on weekdays."]
    assert _support_ratio("The library closes at 20:00 on weekdays.", source) == 1.0


def test_support_ratio_is_zero_for_invented_text():
    source = ["The library closes at 20:00 on weekdays."]
    assert _support_ratio("The chancellor rides a purple helicopter to campus.", source) == 0.0


def test_guardrail_blocks_empty_answer():
    verdict = check(answer_text="", support=0.9, threshold=0.6, retrieved=2)
    assert not verdict.allowed
    assert "no answer" in verdict.reason


def test_guardrail_blocks_when_nothing_retrieved():
    verdict = check(answer_text="Something.", support=1.0, threshold=0.6, retrieved=0)
    assert not verdict.allowed


def test_guardrail_blocks_missing_citation():
    verdict = check(answer_text="Something without a marker.", support=1.0, threshold=0.6, retrieved=1, cited_markers=[])
    assert not verdict.allowed
    assert "citation" in verdict.reason


def test_guardrail_blocks_low_support():
    verdict = check(answer_text="Answer [1].", support=0.2, threshold=0.6, retrieved=1, cited_markers=[1])
    assert not verdict.allowed
    assert "below" in verdict.reason


def test_guardrail_allows_supported_cited_answer():
    verdict = check(answer_text="Answer [1].", support=0.9, threshold=0.6, retrieved=1, cited_markers=[1])
    assert verdict.allowed


def test_extractive_answerer_emits_markers_and_citations(kit):
    result = kit.ask("When does the library close?")
    assert result.allowed
    assert "[1]" in result.answer
    assert result.citations and result.citations[0].chunk_id.startswith("lib-1")


def test_pipeline_refuses_unsupported_question(kit):
    result = kit.ask("Which bus goes to the airport on Sunday morning?")
    assert not result.allowed
    assert result.answer == ""
    assert result.reason


def test_pipeline_result_serialises(kit):
    payload = kit.ask("How many books can an undergraduate borrow?").as_dict()
    assert payload["allowed"] is True
    assert payload["citations"][0]["source"].endswith("library_rules.md")
    assert payload["latency_ms"] >= 0


def test_llm_answerer_handles_not_found(kit):
    provider = FakeProvider("NOT FOUND")
    result = kit.ask("What is the hostel mess menu?", answerer=LLMAnswerer(provider))
    assert not result.allowed
    assert "NOT FOUND" in provider.prompts[0]


def test_llm_answerer_rejects_off_corpus_hallucination(kit):
    provider = FakeProvider("The chancellor rides a purple helicopter every Tuesday [1].")
    result = kit.ask("What is the mess menu?", answerer=LLMAnswerer(provider))
    assert not result.allowed


def test_llm_answerer_prompt_contains_context(kit):
    provider = FakeProvider("NOT FOUND")
    kit.ask("library hours?", answerer=LLMAnswerer(provider))
    assert "[1]" in provider.prompts[0]


def test_stats_reports_index_size(kit):
    stats = kit.stats()
    assert stats["documents"] == 2
    assert stats["chunks"] >= 2


def test_settings_validation():
    try:
        Settings(chunk_size=100, chunk_overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= size")

    try:
        Settings(min_support=1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range min_support")


def test_extractive_answerer_handles_empty_query(kit):
    assert ExtractiveAnswerer().answer("", kit.retrieve("library")).text == ""

class _UncitedProvider:
    """An LLM provider that answers plausibly but cites nothing."""

    name = "uncited-test-double"

    def generate(self, prompt: str) -> str:
        return "The library is open from 08:00 to 20:00 Monday to Friday and closes at 23:30."


def test_answer_without_citation_markers_is_refused(kit):
    """An LLM answer with no ``[n]`` markers must be blocked by the citation gate.

    Regression test: ``cited_markers=answer.cited_markers or None`` turned the empty
    list into ``None``, which the guardrail reads as "no citation check requested",
    so an uncited answer used to be returned to the user.
    """
    from citekit.answering import LLMAnswerer

    answerer = LLMAnswerer(_UncitedProvider())
    result = kit.ask("When does the library close on a Friday?", answerer=answerer)
    assert not result.allowed
    assert result.answer == ""
    assert "citation" in result.reason
