"""Trajectory metrics on hand-built trajectories, including the degenerate
ones: empty, single step, every tool call failed."""
import pytest

from sagwa.adapters.base import AgentRun, AgentStep
from sagwa.metrics.agent_trajectory import compute_trajectory_metrics, cost_per_success, tokens_per_success
from sagwa.storage import Result


def _tool(name, input="x", error=None):
    return AgentStep("tool", name, input=input, output=None if error else "ok", error=error)


def _metrics(steps, expected_tools=("order_db",), tokens=100, latency=50, cost=None):
    run = AgentRun(answer="a", context=None, latency_ms=latency, tokens=tokens, cost_usd=cost, trajectory=steps)
    return compute_trajectory_metrics(run, list(expected_tools) if expected_tools is not None else None)


def test_empty_trajectory_has_no_rates_rather_than_zero_rates():
    m = _metrics([])
    assert m["steps"] == 0 and m["tool_calls"] == 0 and m["handoff_count"] == 0
    for rate in ("tool_call_accuracy", "tool_error_rate", "tool_retry_rate", "loop_rate",
                 "recovery_rate", "handoff_loss_rate"):
        assert m[rate] is None, rate
    assert m["step_error"] is False


def test_single_successful_step():
    m = _metrics([_tool("order_db")])
    assert (m["steps"], m["tool_calls"], m["tool_call_accuracy"], m["tool_error_rate"]) == (1, 1, 1.0, 0.0)
    assert m["loop_rate"] == 0.0 and m["recovery_rate"] is None


def test_tool_call_accuracy_counts_calls_to_unexpected_tools():
    m = _metrics([_tool("order_db"), _tool("corpus_search"), _tool("corpus_search", "y")])
    assert m["tool_call_accuracy"] == pytest.approx(1 / 3)


def test_tool_call_accuracy_is_none_without_expected_tools():
    assert _metrics([_tool("order_db")], expected_tools=None)["tool_call_accuracy"] is None


def test_all_tools_failed():
    m = _metrics([_tool("order_db", "q1", "E"), _tool("order_db", "q2", "E"), _tool("order_db", "q3", "E")])
    assert m["tool_error_rate"] == 1.0
    assert m["tool_retry_rate"] == pytest.approx(2 / 3)  # 2nd and 3rd follow a failure
    assert m["recovery_rate"] == 0.0  # 3 failures, none followed by a success
    assert m["step_error"] is True


def test_recovery_after_a_failed_call():
    m = _metrics([_tool("order_db", "bad", "E"), _tool("order_db", "good")])
    assert m["recovery_rate"] == 1.0 and m["tool_retry_rate"] == 0.5


def test_loop_rate_counts_third_and_later_identical_calls():
    m = _metrics([_tool("order_db", "q"), _tool("order_db", "q"), _tool("order_db", "q"), _tool("order_db", "other")])
    assert m["loop_rate"] == pytest.approx(1 / 4)


def test_one_retry_of_the_same_call_is_not_a_loop():
    """Retrying a failed call once is what the baseline agent is meant to do."""
    m = _metrics([_tool("shipment_tracker", "1050", "503"), _tool("shipment_tracker", "1050", "503")])
    assert m["loop_rate"] == 0.0 and m["tool_retry_rate"] == 0.5


def test_a_retry_loop_is_mostly_loop():
    m = _metrics([_tool("shipment_tracker", "1050", "503") for _ in range(10)])
    assert m["loop_rate"] == pytest.approx(8 / 10)


def test_handoff_loss_rate():
    steps = [AgentStep("handoff", "a->b", input="full findings", output="full findings"),
             AgentStep("handoff", "b->c", input="long text here", output="long")]
    m = _metrics(steps)
    assert m["handoff_count"] == 2 and m["handoff_loss_rate"] == 0.5


def test_economics_are_copied_from_the_run():
    m = _metrics([], tokens=1234, latency=900, cost=0.02)
    assert (m["tokens"], m["latency_ms"], m["cost_usd"]) == (1234, 900, 0.02)


def _result(success, tokens=100, cost=None, error=None):
    return Result(run_id="r", case_id="c", input="", output="", latency_ms=0, tokens=tokens, cost_usd=cost,
                  error=error, metrics_json={} if error else {"task": {"success": success}})


def test_per_success_is_a_ratio_of_run_totals():
    results = [_result(True, 100, 0.1), _result(False, 300, 0.3), _result(True, 200, 0.2)]
    assert tokens_per_success(results) == 300  # 600 tokens / 2 successes
    assert cost_per_success(results) == pytest.approx(0.3)


def test_per_success_ignores_errored_cases_and_is_none_without_data():
    assert tokens_per_success([_result(True, 100), _result(None, None, error="429")]) == 100
    assert tokens_per_success([_result(False, 100)]) is None  # nothing succeeded
    assert cost_per_success([_result(True, 100, cost=None)]) is None  # no cost recorded
    assert cost_per_success([_result(True, 100, 0.1), _result(True, 100, None)]) is None  # partial sum
