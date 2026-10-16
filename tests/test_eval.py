import json
from pathlib import Path

from citekit.config import Settings
from citekit.eval import load_cases, metrics, run
from citekit.loaders import Document, load_directory, load_file, normalize_text
from citekit.pipeline import CiteKit


def test_recall_at_k_counts_relevant_items():
    ranked = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(ranked, {"a", "z"}, 3) == 0.5
    assert metrics.recall_at_k(ranked, set(), 3) == 0.0


def test_mrr_and_ndcg_reward_early_hits():
    ranked = ["x", "rel", "y"]
    assert metrics.mrr_at_k(ranked, {"rel"}, 3) == 0.5
    assert metrics.ndcg_at_k(ranked, {"rel"}, 3) > 0.6
    assert metrics.ndcg_at_k(["rel", "x", "y"], {"rel"}, 3) == 1.0
    assert metrics.mrr_at_k(["x"], {"rel"}, 3) == 0.0


def test_precision_and_hit():
    ranked = ["rel", "a", "b"]
    assert metrics.precision_at_k(ranked, {"rel"}, 3) == 1 / 3
    assert metrics.hit_at_k(ranked, {"rel"}, 2) == 1.0


def test_load_cases_reads_golden_file(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"question": "q1", "must_contain": ["a"]}, {"question": "q2", "answerable": False}]), encoding="utf-8")
    cases = load_cases(path)
    assert [c.question for c in cases] == ["q1", "q2"]
    assert cases[0].must_contain == ("a",)
    assert cases[1].answerable is False


def test_runner_measures_retrieval_and_abstention(kit):
    cases = load_cases(Path("evals/golden.json"))
    report = run(kit, cases, k=5)
    assert report.cases == len(cases)
    assert 0.0 <= report.retrieval["recall@3"] <= 1.0
    assert "abstain_on_unanswerable" in report.answer
    assert report.latency_ms_mean > 0
    assert len(report.rows) == len(cases)
    assert "recall@3" in report.render()


def test_golden_set_abstains_on_unanswerable_questions(kit):
    cases = load_cases(Path("evals/golden.json"))
    report = run(kit, cases, k=5)
    assert report.answer["abstain_on_unanswerable"] == 1.0


def test_normalize_text_strips_html():
    html = "<html><script>evil()</script><p>Hello   world</p></html>"
    out = normalize_text(html, source="page.html")
    assert "Hello world" in out
    assert "evil" not in out


def test_load_file_skips_unsupported_extension(tmp_path):
    path = tmp_path / "notes.rst"
    path.write_text("ignored", encoding="utf-8")
    assert load_file(path) is None


def test_load_directory_and_id_stability(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha content", encoding="utf-8")
    docs = load_directory(tmp_path)
    assert len(docs) == 1
    again = load_directory(tmp_path)
    assert docs[0].id == again[0].id


def test_load_directory_raises_for_file(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("x", encoding="utf-8")
    try:
        load_directory(path)
    except NotADirectoryError:
        return
    raise AssertionError("expected NotADirectoryError")


def test_repo_corpus_answers_and_refuses():
    kit = CiteKit(settings=Settings()).index(load_directory("data/docs"))
    stats = kit.stats()
    assert stats["documents"] >= 4 and stats["chunks"] > stats["documents"]

    answered = kit.ask("What is the monthly bandwidth cap on the secured network?")
    assert answered.allowed and "30 gb" in answered.answer.lower()

    refused = kit.ask("How do I apply for a scholarship for semester fees?")
    assert not refused.allowed