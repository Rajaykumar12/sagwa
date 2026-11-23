"""The trace viewer's query layer and DOT rendering (docs/AGENTIC_PRD.md
Phase 4), without a Streamlit runtime."""
import pytest

from sagwa.dashboard.queries import agentic_runs, load_trace, run_cases, trace_summary
from sagwa.dashboard.trace_view import trace_dot
from sagwa.storage import Result, Run, TraceStep, get_session


def _store():
    with get_session() as session:
        run = Run(sagwa_git_sha="t", target_name="trace-test", model="m", dataset_path="x",
                  dataset_sha256="x", status="completed")
        session.add(run)
        session.flush()
        ok = Result(run_id=run.id, case_id="a-ok", input="q", output="13", latency_ms=90, tokens=300,
                    metrics_json={"task": {"success": True, "failed": []}})
        bad = Result(run_id=run.id, case_id="b-bad", input="q", output="?", latency_ms=900, tokens=500,
                     metrics_json={"task": {"success": False, "failed": ["max_steps"]}})
        session.add_all([ok, bad])
        session.flush()
        # Stored out of order on purpose: the trace must come back by seq.
        session.add_all([
            TraceStep(result_id=bad.id, seq=2, step_type="tool", name="tracker", latency_ms=5,
                      error="ConnectionError: 503", metadata_json={"arguments": {"order_id": "1050"}}),
            TraceStep(result_id=bad.id, seq=0, step_type="agent", name="planner", latency_ms=40, tokens=100,
                      metadata_json={"model": "m-small"}),
            TraceStep(result_id=bad.id, seq=1, step_type="tool", name="tracker", latency_ms=5,
                      error="ConnectionError: 503", metadata_json={}),
            TraceStep(result_id=bad.id, seq=3, step_type="agent", name="writer", latency_ms=30, tokens=80,
                      metadata_json={}),
            TraceStep(result_id=ok.id, seq=0, step_type="agent", name="planner", latency_ms=40, tokens=100,
                      metadata_json={}),
        ])
        return run.id, bad.id


def test_agentic_runs_lists_runs_with_trajectories_only():
    run_id, _ = _store()
    with get_session() as session:
        plain = Run(sagwa_git_sha="t", target_name="no-trace", model="m", dataset_path="x", dataset_sha256="x")
        session.add(plain)
        session.flush()
        session.add(Result(run_id=plain.id, case_id="c", input="q", output="a", latency_ms=1))
        plain_id = plain.id
    with get_session() as session:
        ids = [r["run_id"] for r in agentic_runs(session)]
    assert run_id in ids and plain_id not in ids


def test_run_cases_reports_outcome_and_filters_failures():
    run_id, _ = _store()
    with get_session() as session:
        everything = run_cases(session, run_id)
        failed = run_cases(session, run_id, only_failed=True)
    assert [c["case_id"] for c in everything] == ["a-ok", "b-bad"]
    assert [c["case_id"] for c in failed] == ["b-bad"]
    assert failed[0]["failed"] == ["max_steps"] and failed[0]["steps"] == 4 and failed[0]["step_errors"] == 2


def test_load_trace_returns_steps_in_seq_order_as_plain_dicts():
    _, result_id = _store()
    with get_session() as session:
        steps = load_trace(session, result_id)
    assert [(s["seq"], s["name"]) for s in steps] == [(0, "planner"), (1, "tracker"), (2, "tracker"), (3, "writer")]
    assert steps[2]["metadata"] == {"arguments": {"order_id": "1050"}}
    assert steps[1]["error"].startswith("ConnectionError")


def test_trace_summary_rolls_up_per_agent_and_tool():
    _, result_id = _store()
    with get_session() as session:
        summary = {(r["step_type"], r["name"]): r for r in trace_summary(load_trace(session, result_id))}
    assert summary[("tool", "tracker")] == {"step_type": "tool", "name": "tracker", "steps": 2, "errors": 2,
                                            "latency_ms": 10, "tokens": None, "cost_usd": None}
    assert summary[("agent", "planner")]["tokens"] == 100


def test_trace_summary_of_an_empty_trajectory_is_empty():
    assert trace_summary([]) == []


def test_trace_dot_draws_every_step_in_order_and_marks_failures():
    _, result_id = _store()
    with get_session() as session:
        dot = trace_dot(load_trace(session, result_id))
    assert dot.startswith("digraph trace {") and dot.rstrip().endswith("}")
    assert "s0 -> s1;" in dot and "s1 -> s2;" in dot and "s2 -> s3;" in dot
    failed_nodes = [line for line in dot.splitlines() if "FAILED" in line]
    assert len(failed_nodes) == 2 and all("#f6d5d5" in line for line in failed_nodes)


def test_trace_dot_escapes_quotes_in_labels():
    step = {"seq": 0, "step_type": "tool", "name": 'say "hi"', "latency_ms": None, "tokens": None, "error": None}
    assert 'say \'hi\'' in trace_dot([step])
