from sagwa.metrics.ragas_metrics import _enabled_metrics


def test_defaults_to_both_metrics(monkeypatch):
    monkeypatch.delenv("SAGWA_RAGAS_METRICS", raising=False)
    assert _enabled_metrics() == {"faithfulness", "context_precision"}


def test_env_var_narrows_the_set(monkeypatch):
    monkeypatch.setenv("SAGWA_RAGAS_METRICS", "faithfulness")
    assert _enabled_metrics() == {"faithfulness"}


def test_empty_env_var_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SAGWA_RAGAS_METRICS", "")
    assert _enabled_metrics() == {"faithfulness", "context_precision"}
