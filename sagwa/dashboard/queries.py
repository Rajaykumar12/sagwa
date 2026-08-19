"""Dashboard query layer (PRD FR-26-FR-28) — plain, unit-testable functions
over `Run`/`Result` rows. Kept separate from `app.py` so this logic is
testable without a Streamlit runtime.
"""
from __future__ import annotations

from sagwa.clustering import cluster_run
from sagwa.storage import Result, Run


def _get_metric(result: Result, dotted_path: str):
    value = result.metrics_json or {}
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def metric_trend(session, target_name: str, metric_path: str) -> list[tuple]:
    """One `(created_at, mean_metric_value)` point per `Run` matching
    `target_name`, ordered chronologically (FR-26). Runs with no case
    reporting `metric_path` are excluded, not zeroed."""
    runs = (
        session.query(Run)
        .filter(Run.target_name == target_name)
        .order_by(Run.created_at)
        .all()
    )
    points = []
    for run in runs:
        values = [_get_metric(r, metric_path) for r in run.results]
        values = [float(v) for v in values if v is not None]
        if values:
            points.append((run.created_at, sum(values) / len(values)))
    return points


def cost_latency_trend(session, target_name: str) -> list[tuple]:
    """One `(created_at, mean_cost_usd, mean_latency_ms)` point per `Run`
    (FR-27). `None` cost/latency values are excluded from their mean
    rather than treated as 0."""
    runs = (
        session.query(Run)
        .filter(Run.target_name == target_name)
        .order_by(Run.created_at)
        .all()
    )
    points = []
    for run in runs:
        costs = [r.cost_usd for r in run.results if r.cost_usd is not None]
        latencies = [r.latency_ms for r in run.results if r.error is None]
        mean_cost = sum(costs) / len(costs) if costs else None
        mean_latency = sum(latencies) / len(latencies) if latencies else None
        points.append((run.created_at, mean_cost, mean_latency))
    return points


def run_failure_clusters(session, run_id: str, gates_config: dict):
    """Thin wrapper over `sagwa.clustering.cluster_run` — kept here so
    `app.py` doesn't need to import clustering internals directly."""
    return cluster_run(session, run_id, gates_config=gates_config)


def case_detail(session, run_id: str, case_id: str) -> dict | None:
    """Full detail for one case in one run (FR-28), including judge
    rationale when `sagwa.metrics.judge_metrics` populated it. Returns
    `None` if no matching `Result` exists."""
    result = (
        session.query(Result)
        .filter(Result.run_id == run_id, Result.case_id == case_id)
        .one_or_none()
    )
    if result is None:
        return None
    judge = _get_metric(result, "judge") or {}
    return {
        "case_id": result.case_id,
        "input": result.input,
        "output": result.output,
        "context": result.context,
        "error": result.error,
        "metrics_json": result.metrics_json,
        "judge_score": judge.get("score"),
        "judge_rationale": judge.get("rationale"),
    }
