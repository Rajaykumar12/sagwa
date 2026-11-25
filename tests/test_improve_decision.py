"""Accept/reject for `sagwa improve` against fixture diffs (docs/AGENTIC_PRD.md
Phase 6), including "improved, but a gated metric regressed"."""
from sagwa.diff import DiffResult, MetricDelta
from sagwa.improve.decision import decide

GATES = {"task.success": {"op": "gte", "value": 0.7}, "agent.steps": {"op": "lte", "value": 12},
         "judge.score": {"op": "gte", "value": 0.7}}


def _delta(name, base, cand, significant, test="mcnemar"):
    return MetricDelta(metric_name=name, baseline_mean=base, candidate_mean=cand, delta=cand - base,
                       test=test, p_value=0.01 if significant else 0.4, significant=significant, n=40)


def _diff(*deltas):
    return DiffResult(baseline_run_id="b", candidate_run_id="c", overall=list(deltas), by_tag={}, flips=[],
                      baseline_only_case_ids=[], candidate_only_case_ids=[])


def test_accepts_a_significant_improvement_with_nothing_regressed():
    d = decide(_diff(_delta("task.success", 0.5, 0.8, True), _delta("agent.steps", 8, 8.5, False, "bootstrap_ci")), GATES)
    assert d.accepted and "significant improvement" in d.reasons[0]


def test_rejects_an_improvement_that_isnt_significant():
    """The noise guard: a loop that always 'improves' is measuring noise."""
    d = decide(_diff(_delta("task.success", 0.50, 0.55, False)), GATES)
    assert not d.accepted and "not significant" in d.reasons[0]


def test_rejects_a_significant_change_in_the_wrong_direction():
    d = decide(_diff(_delta("task.success", 0.8, 0.5, True)), GATES)
    assert not d.accepted and d.reasons[0].endswith(": worse")


def test_rejects_an_improvement_when_a_gated_metric_regressed():
    d = decide(_diff(_delta("task.success", 0.5, 0.8, True),
                     _delta("agent.steps", 8.0, 15.0, True, "bootstrap_ci")), GATES)
    assert not d.accepted and d.regressed == ["agent.steps"]
    assert "improved, but a gated metric regressed" in d.reasons[0]


def test_a_significant_change_in_an_ungated_metric_doesnt_block():
    d = decide(_diff(_delta("task.success", 0.5, 0.8, True),
                     _delta("agent.tokens", 2000, 4000, True, "bootstrap_ci")), GATES)
    assert d.accepted


def test_rejects_when_the_primary_metric_is_missing():
    d = decide(_diff(_delta("judge.score", 0.6, 0.9, True)), GATES)
    assert not d.accepted and "not computed in both runs" in d.reasons[0]


def test_lower_is_better_primary_metric():
    d = decide(_diff(_delta("agent.steps", 15, 8, True, "bootstrap_ci")), GATES, primary="agent.steps")
    assert d.accepted


def test_tolerance_ignores_trivially_small_significant_changes():
    d = decide(_diff(_delta("task.success", 0.80, 0.81, True)), GATES, tolerance=0.02)
    assert not d.accepted and d.reasons[0].endswith("within the 0.02 tolerance")
