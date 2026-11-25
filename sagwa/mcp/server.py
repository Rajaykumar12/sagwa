"""Sagwa as an MCP server (PRD FR-40), so a coding agent
can ask "did my change regress the pipeline?" with tool calls alone.

Every tool is a thin wrapper over the function the CLI command calls, and
returns that function's own `to_dict()`. **No evaluation logic lives here**,
so the CLI and MCP answers can never diverge. Paths (datasets, gate configs)
resolve against the server's working directory, like the CLI's.

    sagwa mcp          # stdio; register this command with an MCP client
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("sagwa")


def _tool(fn):
    """Registers `fn` as a tool. Expected failures (an unknown run, a missing
    file, a bad config) become `ToolError`, whose message reaches the calling
    agent; anything else stays a crash whose text stays on the server."""
    from sagwa.datasets import DatasetError
    from sagwa.gate import GateConfigError

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, FileNotFoundError, DatasetError, GateConfigError) as e:
            raise ToolError(str(e)) from e

    return mcp.tool()(wrapper)


def _require_run(session, run_id: str):
    from sagwa.storage import Run

    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"unknown run id: {run_id}")
    return run


def _result(session, run_id: str, case_id: str):
    from sagwa.storage import Result

    result = session.query(Result).filter(Result.run_id == run_id, Result.case_id == case_id).one_or_none()
    if result is None:
        raise ValueError(f"run {run_id} has no case {case_id}")
    return result


@_tool
def run_evaluation(target: str, dataset: str, concurrency: int = 2) -> dict[str, Any]:
    """Run a golden set (JSONL path) against a target adapter ('module.path:ClassName'
    or 'stub') and store every result, like `sagwa run`. Calls the target, and the
    judge/RAGAS when GROQ_API_KEY is set, so it can take minutes and spend tokens."""
    from sagwa.cli import _load_adapter_class, execute_run
    from sagwa.datasets import load_golden_set
    from sagwa.storage import Result, get_session

    run_id, outcomes = execute_run(_load_adapter_class(target)(), target, Path(dataset), load_golden_set(dataset),
                                   concurrency=concurrency)
    with get_session() as session:
        errored = session.query(Result).filter(Result.run_id == run_id, Result.error.isnot(None)).count()
    return {"run_id": run_id, "cases": len(outcomes), "errored": errored}


@_tool
def get_run(run_id: str) -> dict[str, Any]:
    """A stored run: target, model, dataset, prompt versions, status, and case counts."""
    from sagwa.storage import get_session

    with get_session() as session:
        run = _require_run(session, run_id)
        successes = [((r.metrics_json or {}).get("task") or {}).get("success") for r in run.results]
        return {"run_id": run.id, "target": run.target_name, "model": run.model, "dataset": run.dataset_path,
                "prompt_version": run.prompt_version, "status": run.status, "created_at": run.created_at.isoformat(),
                "cases": len(run.results), "errored": sum(1 for r in run.results if r.error),
                "task_success": sum(s is True for s in successes) if any(s is not None for s in successes) else None}


@_tool
def compare_runs(baseline_run_id: str, candidate_run_id: str, gates_config: str = "config/gates.yaml") -> dict[str, Any]:
    """`sagwa diff`: per-metric deltas with paired significance tests, per-tag deltas,
    cases that flipped pass/fail, and the most affected tag. `table` is the CLI's text."""
    from sagwa.diff import diff_runs, format_table
    from sagwa.gate import load_gate_config
    from sagwa.storage import get_session

    gates = load_gate_config(gates_config) if Path(gates_config).exists() else {}
    with get_session() as session:
        _require_run(session, baseline_run_id), _require_run(session, candidate_run_id)
        result = diff_runs(session, baseline_run_id, candidate_run_id, gates_config=gates)
    return {**result.to_dict(), "table": format_table(result)}


@_tool
def check_gate(run_id: str, gates_config: str = "config/gates.yaml", baseline_run_id: str | None = None) -> dict[str, Any]:
    """`sagwa gate`: absolute thresholds and, with a baseline, significant regressions
    against it. `passed` is what a CI check would report."""
    from sagwa.gate import evaluate_gate, load_gate_config, load_regression_config
    from sagwa.storage import get_session

    with get_session() as session:
        _require_run(session, run_id)
        if baseline_run_id:
            _require_run(session, baseline_run_id)
        result = evaluate_gate(session, run_id, load_gate_config(gates_config), baseline_run_id=baseline_run_id,
                               regression_config=load_regression_config(gates_config))
    return result.to_dict()


@_tool
def inspect_failure(run_id: str, case_id: str) -> dict[str, Any]:
    """One case in full: input, output, context, error, every metric (with the judge's
    rationale), and its trajectory roll-up if it is an agent case."""
    from sagwa.dashboard.queries import case_detail, load_trace, trace_summary
    from sagwa.storage import get_session

    with get_session() as session:
        _require_run(session, run_id)
        result_id = _result(session, run_id, case_id).id
        detail = case_detail(session, run_id, case_id)
        steps = load_trace(session, result_id)
    return {**detail, "steps": len(steps), "trace_summary": trace_summary(steps)}


@_tool
def get_trace(run_id: str, case_id: str) -> dict[str, Any]:
    """An agent case's trajectory, step by step: type, name, input, output, latency,
    tokens, error and metadata (tool arguments, model, prompt version)."""
    from sagwa.dashboard.queries import load_trace
    from sagwa.storage import get_session

    with get_session() as session:
        _require_run(session, run_id)
        return {"run_id": run_id, "case_id": case_id, "steps": load_trace(session, _result(session, run_id, case_id).id)}


@_tool
def cluster_failures(run_id: str, gates_config: str = "config/gates.yaml", min_cluster_size: int = 3) -> dict[str, Any]:
    """`sagwa cluster`: failing cases grouped by similarity (trajectory signatures for
    agent runs), each with a label."""
    from sagwa.clustering import cluster_run
    from sagwa.gate import load_gate_config
    from sagwa.storage import get_session

    gates = load_gate_config(gates_config) if Path(gates_config).exists() else {}
    with get_session() as session:
        _require_run(session, run_id)
        clusters = cluster_run(session, run_id, gates_config=gates, min_cluster_size=min_cluster_size)
    return {"run_id": run_id, "clusters": [c.to_dict() for c in clusters]}


@_tool
def root_causes(run_id: str, baseline_run_id: str | None = None) -> dict[str, Any]:
    """`sagwa rootcause`: an agent run's failing cases grouped by cause from a closed
    taxonomy, with evidence. A baseline run adds case-by-case contrast."""
    from sagwa.rootcause import format_report, root_causes as diagnose
    from sagwa.storage import get_session

    with get_session() as session:
        _require_run(session, run_id)
        if baseline_run_id:
            _require_run(session, baseline_run_id)
        causes = diagnose(session, run_id, baseline_run_id=baseline_run_id)
    return {"run_id": run_id, "causes": [c.to_dict() for c in causes], "report": format_report(causes)}


if __name__ == "__main__":
    mcp.run()
