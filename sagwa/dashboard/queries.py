"""Dashboard query layer (PRD FR-26-FR-28) — plain, unit-testable functions
over `Run`/`Result` rows. Kept separate from `app.py` so this logic is
testable without a Streamlit runtime.
"""
from __future__ import annotations

from sagwa.clustering import cluster_run
from sagwa.storage import Result, Run, TraceStep


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


# --- Agent traces (docs/AGENTIC_PRD.md Phase 4) ---------------------------
# Everything below returns plain dicts, not ORM rows: sessions expire their
# objects on commit, and the Streamlit layer reads them after it closes.


def agentic_runs(session) -> list[dict]:
    """Runs that recorded at least one trajectory step, newest first."""
    rows = (
        session.query(Run.id, Run.target_name, Run.created_at)
        .join(Result, Result.run_id == Run.id)
        .join(TraceStep, TraceStep.result_id == Result.id)
        .distinct()
        .order_by(Run.created_at.desc())
        .all()
    )
    return [{"run_id": r.id, "target_name": r.target_name, "created_at": r.created_at} for r in rows]


def run_cases(session, run_id: str, only_failed: bool = False) -> list[dict]:
    """One row per case of a run, with the outcome needed to pick a trace.
    `success` is `task.success` (None for a case without success criteria);
    `only_failed` keeps cases where it is False or the case itself errored."""
    results = session.query(Result).filter(Result.run_id == run_id).order_by(Result.case_id).all()
    rows = []
    for r in results:
        task = _get_metric(r, "task") or {}
        row = {
            "result_id": r.id, "case_id": r.case_id, "success": task.get("success"),
            "failed": task.get("failed", []), "steps": len(r.trace_steps),
            "step_errors": sum(1 for s in r.trace_steps if s.error), "tokens": r.tokens,
            "latency_ms": r.latency_ms, "error": r.error,
        }
        if only_failed and not (row["success"] is False or r.error):
            continue
        rows.append(row)
    return rows


def load_trace(session, result_id: str) -> list[dict]:
    """A case's trajectory in step order (`seq`)."""
    steps = session.query(TraceStep).filter(TraceStep.result_id == result_id).order_by(TraceStep.seq).all()
    return [
        {"seq": s.seq, "step_type": s.step_type, "name": s.name, "input": s.input, "output": s.output,
         "latency_ms": s.latency_ms, "tokens": s.tokens, "cost_usd": s.cost_usd, "error": s.error,
         "metadata": s.metadata_json or {}}
        for s in steps
    ]


def trace_summary(steps: list[dict]) -> list[dict]:
    """Per-agent and per-tool roll-up of a trajectory: how many steps each
    took, and their latency, tokens, cost and errors. In order of first
    appearance. A total of `None` means no step of that kind reported it."""
    rollup: dict[tuple, dict] = {}
    for s in steps:
        key = (s["step_type"], s["name"])
        row = rollup.setdefault(key, {"step_type": key[0], "name": key[1], "steps": 0, "errors": 0,
                                      "latency_ms": None, "tokens": None, "cost_usd": None})
        row["steps"] += 1
        row["errors"] += bool(s["error"])
        for field in ("latency_ms", "tokens", "cost_usd"):
            if s[field] is not None:
                row[field] = (row[field] or 0) + s[field]
    return list(rollup.values())
