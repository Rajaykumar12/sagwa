"""Agent-aware diff and gate (docs/AGENTIC_PRD.md Phase 3), on two synthetic
agentic runs stored the way `sagwa run` stores them."""
from pathlib import Path

import pytest

from sagwa.datasets.schema import GoldenCase, SuccessCriteria
from sagwa.diff import diff_runs, format_table
from sagwa.gate import evaluate_gate, load_gate_config
from sagwa.storage import Result, Run, get_session

AGENTIC_GATES = Path(__file__).parents[1] / "config" / "gates.agentic.yaml"


def _metrics(success, accuracy=1.0, loop=0.0, steps=8, judge=0.8):
    return {"task": {"success": success, "failed": [] if success else ["answer_matches"]},
            "judge": {"score": judge},
            "agent": {"tool_call_accuracy": accuracy, "tool_error_rate": 0.0, "loop_rate": loop,
                      "steps": steps, "tokens": 3000}}


def _store(rows: dict[str, dict]) -> str:
    with get_session() as session:
        run = Run(sagwa_git_sha="t", target_name="agent-test", model="m", dataset_path="unused",
                  dataset_sha256="x", status="completed")
        session.add(run)
        session.flush()
        for case_id, metrics in rows.items():
            session.add(Result(run_id=run.id, case_id=case_id, input="q", output="a", latency_ms=1,
                               tokens=metrics["agent"]["tokens"], metrics_json=metrics))
        return run.id


# 4 database cases and 4 policy cases. The candidate breaks 3 policy cases by
# calling the wrong tool, like V2_ToolOverlap.
CASE_IDS = [f"db-{i}" for i in range(4)] + [f"pol-{i}" for i in range(4)]
CASES = {cid: GoldenCase(id=cid, input="q", task_type="agentic", tags=[cid.split("-")[0]],
                         success_criteria=SuccessCriteria()) for cid in CASE_IDS}


@pytest.fixture
def runs():
    baseline = _store({cid: _metrics(True) for cid in CASE_IDS})
    candidate = _store({cid: _metrics(False, accuracy=0.0) if cid in ("pol-0", "pol-1", "pol-2") else _metrics(True)
                        for cid in CASE_IDS})
    return baseline, candidate


def _diff(baseline, candidate):
    with get_session() as session:
        return diff_runs(session, baseline, candidate, gates_config=load_gate_config(AGENTIC_GATES),
                         golden_cases=CASES)


def test_agent_metrics_flow_into_the_diff_through_the_registry(runs):
    result = _diff(*runs)
    deltas = {m.metric_name: m for m in result.overall}
    assert deltas["task.success"].delta == pytest.approx(-3 / 8) and deltas["task.success"].test == "mcnemar"
    assert deltas["agent.tool_call_accuracy"].delta == pytest.approx(-3 / 8)


def test_flips_and_most_affected_category(runs):
    result = _diff(*runs)
    assert sorted(f.case_id for f in result.flips if f.direction == "pass_to_fail") == ["pol-0", "pol-1", "pol-2"]
    assert result.most_affected_tags == ["pol"]
    assert "Most affected: pol (3 of its cases flipped pass→fail)" in format_table(result)


def test_run_level_metric_is_reported_without_a_significance_test(runs):
    m = {m.metric_name: m for m in _diff(*runs).overall}["tokens.per_success"]
    assert m.test == "run_level" and m.significant is False
    assert m.baseline_mean == 3000 and m.candidate_mean == pytest.approx(24000 / 5)
    row = next(line for line in format_table(_diff(*runs)).splitlines() if line.startswith("tokens.per_success"))
    assert "↑ worse" in row and "n/a" in row  # more tokens per success is worse; no sig test


def test_effect_column_reads_direction_from_the_registry(runs):
    table = format_table(_diff(*runs))
    row = lambda name: next(line for line in table.splitlines() if line.startswith(name))
    assert "↓ worse" in row("task.success")
    assert "same" in row("agent.loop_rate")


def test_agentic_gate_fails_a_looping_run_on_loop_rate():
    looping = _store({cid: _metrics(True, loop=0.8, steps=24) if cid == "db-0" else _metrics(True) for cid in CASE_IDS})
    with get_session() as session:
        result = evaluate_gate(session, looping, load_gate_config(AGENTIC_GATES))
    failed = {m.metric_name for m in result.metric_results if not m.passed}
    assert not result.passed and "agent.loop_rate" in failed
    assert "task.success" not in failed  # a loop can hide behind a correct final answer


def test_agentic_gate_passes_a_baseline_shaped_run(runs):
    with get_session() as session:
        result = evaluate_gate(session, runs[0], load_gate_config(AGENTIC_GATES))
    assert result.passed, [(m.metric_name, m.observed) for m in result.metric_results if not m.passed]
