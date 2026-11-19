"""The metric registry (sagwa/metrics/registry.py) replaced sagwa.diff's
hardcoded metric tuples. These tests pin that the refactor changed nothing, and
guard against the bug class it exists for: a metric that is computed but never
declared, so diff silently drops it."""
import pytest

from sagwa.datasets.schema import GoldenCase
from sagwa.gate import aggregate_metric
from sagwa.metrics import compute_metrics
from sagwa.metrics import registry
from sagwa.metrics.registry import METRICS, MetricSpec, metric_paths
from sagwa.storage import Result

# The tuples sagwa.diff hardcoded before the registry, verbatim.
_OLD_CONTINUOUS = (
    "reference.fuzzy_match",
    "reference.rouge_l_f1",
    "reference.embedding_similarity",
    "classification.precision",
    "classification.recall",
    "classification.f1",
    "ragas.faithfulness",
    "ragas.context_precision",
    "judge.score",
)
_OLD_BINARY = (
    "reference.exact_match",
    "classification.exact_set_match",
    "safety.pii.flagged",
    "safety.toxicity.flagged",
)


def test_registry_split_is_exactly_the_old_tuples_in_the_same_order():
    assert metric_paths("continuous") == _OLD_CONTINUOUS
    assert metric_paths("binary") == _OLD_BINARY


def test_diff_reads_its_paths_from_the_registry():
    from sagwa import diff

    assert diff._CONTINUOUS_METRIC_PATHS == metric_paths("continuous")
    assert diff._BINARY_METRIC_PATHS == metric_paths("binary")


def _numeric_leaves(metrics: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in metrics.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            paths += _numeric_leaves(value, f"{path}.")
        elif isinstance(value, (bool, int, float)) and not key.startswith("_"):
            paths.append(path)
    return paths


@pytest.mark.parametrize(
    "case",
    [
        GoldenCase(id="r", input="Capital of Japan?", expected_output="Tokyo", task_type="rag_qa"),
        GoldenCase(id="c", input="Card charged twice", expected_labels=["billing"], task_type="classification"),
    ],
)
def test_every_numeric_metric_compute_metrics_emits_is_registered(case, monkeypatch):
    """The bug this registry exists for: a metric computed per case but missing
    from diff's list, so `sagwa diff` never reports it."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # judge/RAGAS: no network
    emitted = _numeric_leaves(compute_metrics(case, "Tokyo, billing", context=None))
    assert emitted, "expected some metrics"
    unregistered = [path for path in emitted if path not in METRICS]
    assert not unregistered, f"computed but not in sagwa/metrics/registry.py: {unregistered}"


def _result(metrics_json, cost=None, error=None):
    return Result(run_id="r", case_id="c", input="", output="", latency_ms=0,
                  cost_usd=cost, metrics_json=metrics_json, error=error)


def test_run_level_metric_uses_its_aggregator_not_a_mean(monkeypatch):
    """Cost per successful task is total cost / successful cases, which no
    mean of per-case values can express."""

    def cost_per_success(results):
        successes = sum(1 for r in results if r.metrics_json.get("task", {}).get("success"))
        return sum(r.cost_usd or 0 for r in results) / successes if successes else None

    monkeypatch.setitem(
        registry.METRICS,
        "cost.per_success",
        MetricSpec("continuous", higher_is_better=False, aggregation="run_level", aggregator=cost_per_success),
    )
    results = [
        _result({"task": {"success": True}}, cost=0.30),
        _result({"task": {"success": False}}, cost=0.50),
        _result({"task": {"success": True}}, cost=0.20),
    ]
    assert aggregate_metric(results, "cost.per_success") == pytest.approx(0.50)  # 1.00 / 2


def test_run_level_metrics_are_left_out_of_diffs_paired_tests(monkeypatch):
    monkeypatch.setitem(
        registry.METRICS,
        "cost.per_success",
        MetricSpec("continuous", higher_is_better=False, aggregation="run_level", aggregator=lambda rs: None),
    )
    assert "cost.per_success" not in metric_paths("continuous")


def test_unregistered_path_is_still_averaged_by_the_gate():
    results = [_result({"custom": {"x": 1.0}}), _result({"custom": {"x": 3.0}})]
    assert aggregate_metric(results, "custom.x") == 2.0
