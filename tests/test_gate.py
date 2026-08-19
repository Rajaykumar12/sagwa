from pathlib import Path

import pytest
import yaml

from sagwa.gate import (
    GateConfigError,
    _compare,
    aggregate_metric,
    evaluate_gate,
    load_gate_config,
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
