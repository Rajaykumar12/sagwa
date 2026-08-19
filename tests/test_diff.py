import pytest

from sagwa.diff import bootstrap_ci, case_passes, diff_runs, mcnemar_exact
from sagwa.datasets.schema import GoldenCase
from sagwa.storage import Result, Run, get_session


def test_bootstrap_ci_identical_arrays_not_significant():
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    ci_low, ci_high, significant = bootstrap_ci(values, values, n_resamples=500, seed=1)
    assert ci_low <= 0 <= ci_high
    assert significant is False


def test_bootstrap_ci_large_effect_is_significant():
    baseline = [0.1] * 20
    candidate = [0.9] * 20
    ci_low, ci_high, significant = bootstrap_ci(baseline, candidate, n_resamples=500, seed=1)
    assert ci_low > 0
    assert significant is True


def test_mcnemar_exact_known_fixture():
    # 3 cases flip pass->fail (True->False), 1 flips fail->pass, rest agree.
    baseline = [True, True, True, False, True, True]
    candidate = [False, False, False, True, True, True]
    p_value, significant = mcnemar_exact(baseline, candidate)
    assert 0.0 <= p_value <= 1.0
    assert isinstance(significant, bool)


def test_mcnemar_exact_no_discordant_pairs_is_not_significant():
    values = [True, False, True]
    p_value, significant = mcnemar_exact(values, values)
    assert p_value == 1.0
    assert significant is False


def _make_result(run_id, case_id, metrics_json=None, error=None):
    return Result(
        run_id=run_id,
        case_id=case_id,
        input="x",
        output="" if error else "y",
        latency_ms=1,
        metrics_json=metrics_json or {},
        error=error,
    )


def test_case_passes_reflects_gate_thresholds():
    gates = {"reference.exact_match": {"op": "gte", "value": 1.0}}
    passing = _make_result("r", "1", {"reference": {"exact_match": 1.0}})
    failing = _make_result("r", "2", {"reference": {"exact_match": 0.0}})
    no_data = _make_result("r", "3", {"reference": {}})
    errored = _make_result("r", "4", {}, error="boom")

    assert case_passes(passing, gates) is True
    assert case_passes(failing, gates) is False
    assert case_passes(no_data, gates) is None
    assert case_passes(errored, gates) is None


def test_diff_runs_end_to_end():
    with get_session() as session:
        baseline_run = Run(
            sagwa_git_sha="a", target_name="stub", model="n/a",
            dataset_path="golden_sets/example.jsonl", dataset_sha256="0" * 64,
        )
        candidate_run = Run(
            sagwa_git_sha="b", target_name="stub", model="n/a",
            dataset_path="golden_sets/example.jsonl", dataset_sha256="0" * 64,
        )
        session.add_all([baseline_run, candidate_run])
        session.flush()
        baseline_id, candidate_id = baseline_run.id, candidate_run.id

        # case-1: passes in both. case-2: flips pass->fail. case-3: baseline only.
        session.add(_make_result(baseline_id, "case-1", {"reference": {"exact_match": 1.0, "fuzzy_match": 0.9}}))
        session.add(_make_result(baseline_id, "case-2", {"reference": {"exact_match": 1.0, "fuzzy_match": 0.9}}))
        session.add(_make_result(baseline_id, "case-3", {"reference": {"exact_match": 1.0, "fuzzy_match": 0.9}}))

        session.add(_make_result(candidate_id, "case-1", {"reference": {"exact_match": 1.0, "fuzzy_match": 0.95}}))
        session.add(_make_result(candidate_id, "case-2", {"reference": {"exact_match": 0.0, "fuzzy_match": 0.1}}))
        session.add(_make_result(candidate_id, "case-4", {"reference": {"exact_match": 1.0, "fuzzy_match": 0.9}}))

    try:
        gates = {"reference.exact_match": {"op": "gte", "value": 1.0}}
        golden_cases = {
            "case-1": GoldenCase(id="case-1", input="x", task_type="rag_qa", tags=["easy"]),
            "case-2": GoldenCase(id="case-2", input="x", task_type="rag_qa", tags=["hard"]),
            "case-4": GoldenCase(id="case-4", input="x", task_type="rag_qa", tags=["easy"]),
        }
        with get_session() as session:
            result = diff_runs(session, baseline_id, candidate_id, gates_config=gates, golden_cases=golden_cases)

        assert result.baseline_only_case_ids == ["case-3"]
        assert result.candidate_only_case_ids == ["case-4"]

        assert len(result.flips) == 1
        assert result.flips[0].case_id == "case-2"
        assert result.flips[0].direction == "pass_to_fail"
        assert result.flips[0].tags == ["hard"]

        exact_match_delta = next(m for m in result.overall if m.metric_name == "reference.exact_match")
        assert exact_match_delta.baseline_mean == pytest.approx(1.0)
        assert exact_match_delta.candidate_mean == pytest.approx(0.5)
        assert exact_match_delta.test == "mcnemar"

        fuzzy_delta = next(m for m in result.overall if m.metric_name == "reference.fuzzy_match")
        assert fuzzy_delta.test == "bootstrap_ci"

        assert "easy" in result.by_tag
        assert "hard" in result.by_tag

        import json

        json.dumps(result.to_dict())  # round-trips without raising
    finally:
        with get_session() as session:
            for run_id in (baseline_id, candidate_id):
                run_row = session.get(Run, run_id)
                if run_row is not None:
                    for r in list(run_row.results):
                        session.delete(r)
                    session.delete(run_row)
