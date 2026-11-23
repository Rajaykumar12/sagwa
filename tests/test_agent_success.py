"""`task.success`: each SuccessCriteria condition, positive and negative."""
import pytest

from sagwa.adapters.base import AgentRun, AgentStep
from sagwa.datasets.schema import SuccessCriteria
from sagwa.metrics.agent_success import check_success, compute_success


def _run(answer="There were 13 late orders.", steps=None):
    steps = steps if steps is not None else [AgentStep("agent", "planner"), AgentStep("tool", "order_db"),
                                              AgentStep("agent", "writer")]
    return AgentRun(answer=answer, context=None, latency_ms=1, tokens=1, cost_usd=None, trajectory=steps)


@pytest.mark.parametrize("criteria, run, failed", [
    (SuccessCriteria(answer_contains=["13 LATE"]), _run(), []),                   # case-insensitive
    (SuccessCriteria(answer_contains=["14"]), _run(), ["answer_contains"]),
    (SuccessCriteria(answer_matches=r"\b13\b"), _run(), []),
    (SuccessCriteria(answer_matches=r"\b113\b"), _run(), ["answer_matches"]),
    (SuccessCriteria(required_tools=["order_db"]), _run(), []),
    (SuccessCriteria(required_tools=["order_db", "calculator"]), _run(), ["required_tools"]),
    (SuccessCriteria(forbidden_tools=["calculator"]), _run(), []),
    (SuccessCriteria(forbidden_tools=["order_db"]), _run(), ["forbidden_tools"]),
    (SuccessCriteria(max_steps=3), _run(), []),
    (SuccessCriteria(max_steps=2), _run(), ["max_steps"]),
])
def test_each_condition(criteria, run, failed):
    assert check_success(criteria, run) == failed


def test_a_step_error_fails_by_default_and_passes_when_allowed():
    run = _run(steps=[AgentStep("tool", "order_db", error="OperationalError: no such column"),
                      AgentStep("agent", "writer")])
    assert check_success(SuccessCriteria(), run) == ["must_not_error"]
    assert check_success(SuccessCriteria(must_not_error=False), run) == []


def test_every_failed_condition_is_reported_not_just_the_first():
    criteria = SuccessCriteria(answer_contains=["14"], forbidden_tools=["order_db"], max_steps=1)
    assert compute_success(criteria, _run()) == {
        "success": False, "failed": ["answer_contains", "forbidden_tools", "max_steps"]}


def test_empty_criteria_with_a_clean_run_is_success():
    assert compute_success(SuccessCriteria(), _run()) == {"success": True, "failed": []}


def test_empty_trajectory_and_answer():
    run = _run(answer="", steps=[])
    assert check_success(SuccessCriteria(answer_contains=["13"]), run) == ["answer_contains"]
    assert check_success(SuccessCriteria(required_tools=["order_db"]), run) == ["required_tools"]


def test_unicode_spaces_and_dashes_dont_fail_a_correct_answer():
    """Seen in the first full agent run: "Elena\u202fRossi", "5\u202fyears"."""
    run = _run(answer="Elena\u202fRossi has a 5\u202fyear warranty, from 2026\u201110\u201101.")
    assert check_success(SuccessCriteria(answer_contains=["Elena Rossi"]), run) == []
    assert check_success(SuccessCriteria(answer_matches=r"\b5[- ]year"), run) == []
    assert check_success(SuccessCriteria(answer_contains=["2026-10-01"]), run) == []
