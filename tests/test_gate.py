from pathlib import Path

import pytest
import yaml

from sagwa.gate import (
    GateConfigError,
    _compare,
    _is_worse,
    aggregate_metric,
    evaluate_gate,
    load_gate_config,
    load_regression_config,
)
from sagwa.storage import Result, Run, get_session


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_gate_config_round_trips(tmp_path):
    path = _write_config(tmp_path, {"metrics": {"faithfulness": {"op": "gte", "value": 0.85}}})
    config = load_gate_config(path)
    assert config == {"faithfulness": {"op": "gte", "value": 0.85}}


def test_load_gate_config_missing_file_raises(tmp_path):
    with pytest.raises(GateConfigError):
        load_gate_config(tmp_path / "does_not_exist.yaml")


def test_load_gate_config_missing_metrics_section_raises(tmp_path):
    path = _write_config(tmp_path, {"other": {}})
    with pytest.raises(GateConfigError):
        load_gate_config(path)


def test_load_gate_config_invalid_op_raises(tmp_path):
    path = _write_config(tmp_path, {"metrics": {"faithfulness": {"op": "nope", "value": 0.5}}})
    with pytest.raises(GateConfigError):
        load_gate_config(path)


def test_compare_all_ops_and_boundaries():
    assert _compare(0.85, "gte", 0.85) is True
    assert _compare(0.85, "gt", 0.85) is False
    assert _compare(0.5, "lte", 0.5) is True
    assert _compare(0.5, "lt", 0.5) is False


def test_compare_unknown_op_raises():
    with pytest.raises(ValueError):
        _compare(0.5, "eq", 0.5)


def _make_result(run_id, case_id, metrics_json, error=None):
    return Result(
        run_id=run_id,
        case_id=case_id,
        input="x",
        output="" if error else "y",
        latency_ms=1,
        metrics_json=metrics_json,
        error=error,
    )


def test_aggregate_metric_mean_excludes_errors_and_absent():
    results = [
        _make_result("r", "1", {"ragas": {"faithfulness": 0.8}}),
        _make_result("r", "2", {"ragas": {"faithfulness": 1.0}}),
        _make_result("r", "3", {"ragas": {"faithfulness": 0.6}}, error="boom"),
        _make_result("r", "4", {"reference": {}}),
    ]
    assert aggregate_metric(results, "ragas.faithfulness") == pytest.approx(0.9)


def test_aggregate_metric_returns_none_when_absent_everywhere():
    results = [_make_result("r", "1", {"reference": {}})]
    assert aggregate_metric(results, "ragas.faithfulness") is None


def test_evaluate_gate_end_to_end():
    with get_session() as session:
        run_row = Run(
            sagwa_git_sha="deadbeef",
            target_name="stub",
            model="n/a",
            dataset_path="golden_sets/example.jsonl",
            dataset_sha256="0" * 64,
        )
        session.add(run_row)
        session.flush()
        run_id = run_row.id
        session.add(_make_result(run_id, "1", {"ragas": {"faithfulness": 0.9, "context_precision": 0.5}}))
        session.add(_make_result(run_id, "2", {"ragas": {"faithfulness": 0.95, "context_precision": 0.6}}))

    try:
        config = {
            "ragas.faithfulness": {"op": "gte", "value": 0.85},
            "ragas.context_precision": {"op": "gte", "value": 0.8},
        }
        with get_session() as session:
            result = evaluate_gate(session, run_id, config)

        assert result.passed is False
        by_name = {m.metric_name: m for m in result.metric_results}
        assert by_name["ragas.faithfulness"].passed is True
        assert by_name["ragas.context_precision"].passed is False

        # a configured-but-never-computed metric fails loudly, not skipped
        config["missing.metric"] = {"op": "gte", "value": 0.5}
        with get_session() as session:
            result2 = evaluate_gate(session, run_id, config)
        by_name2 = {m.metric_name: m for m in result2.metric_results}
        assert by_name2["missing.metric"].observed is None
        assert by_name2["missing.metric"].passed is False

        markdown = result.to_markdown()
        assert run_id in markdown
        assert "ragas.faithfulness" in markdown
    finally:
        with get_session() as session:
            run_row = session.get(Run, run_id)
            if run_row is not None:
                for r in list(run_row.results):
                    session.delete(r)
                session.delete(run_row)


def test_aggregate_metric_skips_nan_values():
    # RAGAS returns NaN for a case it can't score; one NaN must not poison
    # the run's mean (real HotpotQA run, 2026-09-20).
    results = [
        _make_result("r", "1", {"ragas": {"faithfulness": 0.8}}),
        _make_result("r", "2", {"ragas": {"faithfulness": float("nan")}}),
        _make_result("r", "3", {"ragas": {"faithfulness": 1.0}}),
    ]
    assert aggregate_metric(results, "ragas.faithfulness") == pytest.approx(0.9)


def test_aggregate_metric_is_none_when_every_value_is_nan():
    results = [_make_result("r", "1", {"ragas": {"faithfulness": float("nan")}})]
    assert aggregate_metric(results, "ragas.faithfulness") is None


# --- baseline-relative gating (PRD G2 + G4) -----------------------------
# Absolute thresholds cannot express "this PR is worse than main". These
# cover the half that can: a metric fails when its change from the baseline
# is BOTH statistically significant and in the bad direction.


def _make_run_with(session, values_by_metric: dict[str, list[float]]) -> str:
    """Creates a run whose case i has each metric's i-th value."""
    run_row = Run(
        sagwa_git_sha="deadbeef",
        target_name="stub",
        model="n/a",
        dataset_path="golden_sets/example.jsonl",
        dataset_sha256="0" * 64,
    )
    session.add(run_row)
    session.flush()
    n = len(next(iter(values_by_metric.values())))
    for i in range(n):
        metrics: dict = {}
        for dotted, values in values_by_metric.items():
            *sections, key = dotted.split(".")  # e.g. safety.toxicity.flagged nests twice
            node = metrics
            for section in sections:
                node = node.setdefault(section, {})
            node[key] = values[i]
        session.add(_make_result(run_row.id, str(i), metrics))
    return run_row.id


def test_is_worse_uses_the_metrics_own_op_for_direction():
    # gte/gt: higher is better, so a drop is the regression.
    assert _is_worse("m", -0.2, "gte", tolerance=0.0) is True
    assert _is_worse("m", +0.2, "gte", tolerance=0.0) is False
    # lte/lt (safety flags, error rates): the reverse.
    assert _is_worse("m", +0.2, "lte", tolerance=0.0) is True
    assert _is_worse("m", -0.2, "lte", tolerance=0.0) is False


def test_is_worse_respects_tolerance():
    """With enough cases a trivially small drop is still significant —
    tolerance is what stops that failing a build."""
    assert _is_worse("m", -0.01, "gte", tolerance=0.02) is False
    assert _is_worse("m", -0.05, "gte", tolerance=0.02) is True


def test_load_regression_config_absent_section_returns_empty(tmp_path):
    """No `regression:` section means the gate behaves exactly as before."""
    path = _write_config(tmp_path, {"metrics": {"judge.score": {"op": "gte", "value": 0.6}}})
    assert load_regression_config(path) == {}


def test_load_regression_config_reads_tolerance(tmp_path):
    path = _write_config(
        tmp_path,
        {"metrics": {"judge.score": {"op": "gte", "value": 0.6}}, "regression": {"tolerance": 0.05}},
    )
    assert load_regression_config(path) == {"tolerance": 0.05}


def test_load_regression_config_rejects_non_mapping(tmp_path):
    path = _write_config(
        tmp_path, {"metrics": {"judge.score": {"op": "gte", "value": 0.6}}, "regression": "nope"}
    )
    with pytest.raises(GateConfigError):
        load_regression_config(path)


def test_significant_drop_fails_the_regression_check():
    with get_session() as session:
        baseline_id = _make_run_with(session, {"judge.score": [0.9] * 20})
        candidate_id = _make_run_with(session, {"judge.score": [0.3] * 20})

    config = {"judge.score": {"op": "gte", "value": 0.0}}  # absolute gate passes
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    assert result.passed is False, "a large, consistent drop must fail the gate"
    assert all(m.passed for m in result.metric_results), "the absolute half should still pass"
    regression = {m.metric_name: m for m in result.regression_results}
    assert regression["judge.score vs baseline"].passed is False
    assert regression["judge.score vs baseline"].observed == pytest.approx(-0.6)


def test_noise_level_change_passes_the_regression_check():
    values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.52, 0.58, 0.62, 0.68, 0.71]
    jittered = [v + (0.01 if i % 2 else -0.01) for i, v in enumerate(values)]
    with get_session() as session:
        baseline_id = _make_run_with(session, {"judge.score": values})
        candidate_id = _make_run_with(session, {"judge.score": jittered})

    config = {"judge.score": {"op": "gte", "value": 0.0}}
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    assert result.passed is True


def test_lower_is_better_metric_fails_when_it_rises():
    with get_session() as session:
        baseline_id = _make_run_with(session, {"safety.toxicity.flagged": [0.0] * 20})
        candidate_id = _make_run_with(session, {"safety.toxicity.flagged": [1.0] * 20})

    config = {"safety.toxicity.flagged": {"op": "lte", "value": 1.0}}  # absolute gate passes
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    regression = {m.metric_name: m for m in result.regression_results}
    assert regression["safety.toxicity.flagged vs baseline"].passed is False


def test_metric_the_diff_engine_does_not_know_gets_no_regression_row():
    """A documented limitation, not an oversight: regression checking reuses
    the diff engine, which computes deltas only for the metric paths it knows
    (`sagwa.diff._CONTINUOUS_METRIC_PATHS` / `_BINARY_METRIC_PATHS`). Gate a
    custom path and the absolute threshold still applies, but nothing compares
    it against the baseline."""
    with get_session() as session:
        baseline_id = _make_run_with(session, {"custom.metric": [0.9] * 20})
        candidate_id = _make_run_with(session, {"custom.metric": [0.1] * 20})

    config = {"custom.metric": {"op": "gte", "value": 0.0}}
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    assert result.regression_results == []
    assert result.passed is True  # only the absolute threshold applied


def test_improvement_passes_even_when_significant():
    with get_session() as session:
        baseline_id = _make_run_with(session, {"judge.score": [0.3] * 20})
        candidate_id = _make_run_with(session, {"judge.score": [0.9] * 20})

    config = {"judge.score": {"op": "gte", "value": 0.0}}
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    assert result.passed is True
    assert result.regression_results[0].observed == pytest.approx(0.6)


def test_without_baseline_behaviour_is_unchanged():
    """The whole point of the optional flag: existing users see no change."""
    with get_session() as session:
        run_id = _make_run_with(session, {"judge.score": [0.9] * 5})

    config = {"judge.score": {"op": "gte", "value": 0.6}}
    with get_session() as session:
        result = evaluate_gate(session, run_id, config)

    assert result.passed is True
    assert result.regression_results == []
    assert result.baseline_run_id is None
    assert "vs baseline" not in result.to_markdown()


def test_metric_missing_from_one_run_is_left_to_the_absolute_gate():
    with get_session() as session:
        baseline_id = _make_run_with(session, {"judge.score": [0.9] * 5})
        candidate_id = _make_run_with(session, {"reference.fuzzy_match": [0.9] * 5})

    config = {"judge.score": {"op": "gte", "value": 0.6}}
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    # Nothing to compare — but the absolute gate still fails it loudly for
    # having no judge.score at all, which is the existing contract.
    assert result.regression_results == []
    assert result.passed is False


def test_markdown_and_dict_carry_the_regression_section():
    with get_session() as session:
        baseline_id = _make_run_with(session, {"judge.score": [0.9] * 20})
        candidate_id = _make_run_with(session, {"judge.score": [0.3] * 20})

    config = {"judge.score": {"op": "gte", "value": 0.0}}
    with get_session() as session:
        result = evaluate_gate(session, candidate_id, config, baseline_run_id=baseline_id)

    markdown = result.to_markdown()
    assert f"vs baseline `{baseline_id}`" in markdown
    assert "-0.600" in markdown
    payload = result.to_dict()
    assert payload["baseline_run_id"] == baseline_id
    assert payload["regression_results"][0]["passed"] is False
