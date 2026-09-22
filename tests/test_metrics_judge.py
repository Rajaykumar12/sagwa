import os

from sagwa.datasets.schema import GoldenCase
from sagwa.metrics import compute_metrics
from sagwa.metrics.judge_metrics import compute_judge_metric


def test_compute_judge_metric_returns_none_without_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert compute_judge_metric(query="q", answer="a") is None


def test_compute_metrics_omits_judge_key_without_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    case = GoldenCase(id="c1", input="What is the capital of France?", task_type="rag_qa", tags=[])
    metrics = compute_metrics(case, answer="Paris", context=None)
    assert "judge" not in metrics


def test_compute_judge_metric_degrades_on_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    import sagwa.metrics.judge_metrics as judge_metrics

    monkeypatch.setattr(judge_metrics, "_llm_call", None)

    def _broken_groq_llm_call(*args, **kwargs):
        raise RuntimeError("network unavailable in test")

    monkeypatch.setattr("sagwa.judge.harness.groq_llm_call", _broken_groq_llm_call)

    result = compute_judge_metric(query="q", answer="a")
    assert result == {"score": None, "rationale": None, "_error": "network unavailable in test"}


def _fake_judge(monkeypatch, captured: dict):
    """Installs a judge that records what it was asked to score."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    import sagwa.metrics.judge_metrics as judge_metrics

    monkeypatch.setattr(judge_metrics, "_llm_call", None)
    monkeypatch.setattr("sagwa.judge.harness.groq_llm_call", lambda *a, **k: (lambda prompt: "0.5"))

    def _score(llm_call, query, answer, expected=None, context=None, **kwargs):
        captured.update(query=query, answer=answer, expected=expected, context=context)
        return 0.5, "0.5"

    monkeypatch.setattr("sagwa.judge.harness.score_absolute_with_rationale", _score)


def test_judge_metric_records_the_prompt_version(monkeypatch):
    """A judge score is only comparable to others from the same prompt, and
    only trustworthy up to the kappa measured for that version (FR-14)."""
    from sagwa.judge.harness import JUDGE_PROMPT_VERSION

    _fake_judge(monkeypatch, {})
    result = compute_judge_metric(query="q", answer="a")
    assert result["prompt_version"] == JUDGE_PROMPT_VERSION


def test_judge_metric_forwards_expected_answer_and_context(monkeypatch):
    captured = {}
    _fake_judge(monkeypatch, captured)
    compute_judge_metric(query="q", answer="a", expected="gold", context="ctx")
    assert captured["expected"] == "gold"
    assert captured["context"] == "ctx"


def test_compute_metrics_passes_the_golden_case_reference_to_the_judge(monkeypatch):
    """The wiring that matters: without it the judge grades fluency alone and
    rewards confident wrong answers (docs/STATUS.md §2a)."""
    captured = {}
    _fake_judge(monkeypatch, captured)
    # Stub RAGAS: a rag_qa case with context would otherwise make real network
    # calls, which took 4.5 minutes of retries against a fake key.
    monkeypatch.setattr(
        "sagwa.metrics.ragas_metrics.compute_ragas_metrics", lambda **kwargs: {"faithfulness": 1.0}
    )
    case = GoldenCase(id="c1", input="Who directed Jaws?", expected_output="Steven Spielberg", task_type="rag_qa")
    compute_metrics(case, answer="Spielberg", context="Jaws is a 1975 film directed by Steven Spielberg.")
    assert captured["expected"] == "Steven Spielberg"
    assert "1975 film" in captured["context"]
