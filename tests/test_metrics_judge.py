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
