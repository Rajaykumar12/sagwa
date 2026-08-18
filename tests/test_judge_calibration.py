"""Tests for the calibration math and artifact/refusal plumbing against a
small, deliberately-synthetic fixture (10 labels). This validates the
*mechanism* only — it is not a real calibration result and must never be
cited as one. The real PRD-required calibration (~150-200 hand-labeled
cases, FR-13) is tracked separately in docs/GAP_ANALYSIS.md.
"""
import pytest

from sagwa.judge.calibration import (
    CalibrationRequired,
    calibrate,
    cohen_kappa,
    confusion_matrix,
    load_calibration_results,
    require_calibration,
    save_calibration_result,
)

# Synthetic fixture: 8/10 agreement between "human" and "judge" labels.
_SYNTHETIC_HUMAN_LABELS = ["pass", "pass", "fail", "fail", "pass", "fail", "pass", "fail", "pass", "fail"]
_SYNTHETIC_JUDGE_LABELS = ["pass", "pass", "fail", "pass", "pass", "fail", "pass", "fail", "fail", "fail"]


def test_cohen_kappa_perfect_agreement_is_one():
    labels = ["pass", "fail", "pass", "fail"]
    assert cohen_kappa(labels, labels) == 1.0


def test_cohen_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa(["pass"], ["pass", "fail"])


def test_cohen_kappa_on_synthetic_fixture_is_between_zero_and_one():
    kappa = cohen_kappa(_SYNTHETIC_HUMAN_LABELS, _SYNTHETIC_JUDGE_LABELS)
    assert 0.0 < kappa < 1.0


def test_confusion_matrix_counts_add_up_to_n():
    matrix = confusion_matrix(_SYNTHETIC_HUMAN_LABELS, _SYNTHETIC_JUDGE_LABELS)
    assert sum(matrix.values()) == len(_SYNTHETIC_HUMAN_LABELS)
    assert matrix["pass|pass"] == 4


def test_calibrate_computes_accuracy_and_kappa():
    result = calibrate(_SYNTHETIC_HUMAN_LABELS, _SYNTHETIC_JUDGE_LABELS, judge_prompt_version="v0-synthetic-test")
    assert result.n == 10
    assert result.accuracy == 0.8
    assert result.judge_prompt_version == "v0-synthetic-test"


def test_calibrate_rejects_empty_input():
    with pytest.raises(ValueError):
        calibrate([], [], judge_prompt_version="v0")


def test_save_and_load_calibration_result_roundtrip(tmp_path):
    result = calibrate(_SYNTHETIC_HUMAN_LABELS, _SYNTHETIC_JUDGE_LABELS, judge_prompt_version="v0-synthetic-test")
    path = save_calibration_result(result, directory=tmp_path)
    assert path.exists()

    loaded = load_calibration_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].judge_prompt_version == "v0-synthetic-test"
    assert loaded[0].accuracy == result.accuracy


def test_require_calibration_raises_when_none_meets_threshold(tmp_path):
    result = calibrate(_SYNTHETIC_HUMAN_LABELS, _SYNTHETIC_JUDGE_LABELS, judge_prompt_version="v0-synthetic-test")
    save_calibration_result(result, directory=tmp_path)

    with pytest.raises(CalibrationRequired):
        require_calibration("v0-synthetic-test", min_kappa=0.99, directory=tmp_path)


def test_require_calibration_raises_when_no_record_exists(tmp_path):
    with pytest.raises(CalibrationRequired):
        require_calibration("nonexistent-version", directory=tmp_path)


def test_require_calibration_returns_result_when_threshold_met(tmp_path):
    perfect_labels = ["pass", "fail", "pass", "fail", "pass"]
    result = calibrate(perfect_labels, perfect_labels, judge_prompt_version="v0-perfect")
    save_calibration_result(result, directory=tmp_path)

    found = require_calibration("v0-perfect", min_kappa=0.70, directory=tmp_path)
    assert found.cohen_kappa == 1.0


def test_baseline_comparison_mode_records_baseline_name(tmp_path):
    """FR-15a: scoring a prior judge (e.g. ringo's eval.py) against the same
    human-labeled set is just `calibrate(..., baseline_name=...)`."""
    result = calibrate(
        _SYNTHETIC_HUMAN_LABELS,
        _SYNTHETIC_JUDGE_LABELS,
        judge_prompt_version="ringo_eval_py_baseline",
        baseline_name="ringo_eval_py",
    )
    path = save_calibration_result(result, directory=tmp_path)
    assert "ringo_eval_py" in path.name
    assert load_calibration_results(tmp_path)[0].baseline_name == "ringo_eval_py"
