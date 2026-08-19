import pytest

from sagwa.dashboard.queries import case_detail, cost_latency_trend, metric_trend
from sagwa.storage import Result, Run, get_session


def _make_run(target_name, dataset_sha="0" * 64):
    return Run(
        sagwa_git_sha="a", target_name=target_name, model="n/a",
        dataset_path="golden_sets/example.jsonl", dataset_sha256=dataset_sha,
    )


def _make_result(run_id, case_id, metrics_json=None, cost_usd=None, latency_ms=10, error=None):
    return Result(
        run_id=run_id, case_id=case_id, input="x", output="" if error else "y",
        latency_ms=latency_ms, cost_usd=cost_usd, metrics_json=metrics_json or {}, error=error,
    )


def test_metric_trend_and_cost_latency_trend():
    with get_session() as session:
        run_a = _make_run("target-x")
        run_b = _make_run("target-y")
        session.add_all([run_a, run_b])
        session.flush()
        run_a_id, run_b_id = run_a.id, run_b.id

        session.add(_make_result(run_a_id, "1", {"reference": {"fuzzy_match": 0.8}}, cost_usd=0.01, latency_ms=100))
        session.add(_make_result(run_a_id, "2", {"reference": {"fuzzy_match": 1.0}}, cost_usd=0.02, latency_ms=200))
        session.add(_make_result(run_a_id, "3", {"reference": {}}, cost_usd=None, latency_ms=300))
        # different target — must not be included when filtering on target-x
        session.add(_make_result(run_b_id, "4", {"reference": {"fuzzy_match": 0.0}}, cost_usd=0.5, latency_ms=1))

    try:
        with get_session() as session:
            points = metric_trend(session, "target-x", "reference.fuzzy_match")
            assert len(points) == 1
            assert points[0][1] == pytest.approx(0.9)  # mean of 0.8, 1.0 — case "3" excluded (no data)

            cl_points = cost_latency_trend(session, "target-x")
            assert len(cl_points) == 1
            _, mean_cost, mean_latency = cl_points[0]
            assert mean_cost == pytest.approx(0.015)  # mean of 0.01, 0.02 — None excluded, not zeroed
            assert mean_latency == pytest.approx(200)  # mean of 100, 200, 300
    finally:
        with get_session() as session:
            for run_id in (run_a_id, run_b_id):
                run_row = session.get(Run, run_id)
                if run_row is not None:
                    for r in list(run_row.results):
                        session.delete(r)
                    session.delete(run_row)


def test_case_detail_includes_judge_rationale_when_present():
    with get_session() as session:
        run_row = _make_run("target-z")
        session.add(run_row)
        session.flush()
        run_id = run_row.id
        session.add(
            _make_result(
                run_id, "case-1",
                {"judge": {"score": 0.7, "rationale": "mostly correct"}},
            )
        )

    try:
        with get_session() as session:
            detail = case_detail(session, run_id, "case-1")
        assert detail is not None
        assert detail["judge_score"] == 0.7
        assert detail["judge_rationale"] == "mostly correct"

        with get_session() as session:
            missing = case_detail(session, run_id, "no-such-case")
        assert missing is None
    finally:
        with get_session() as session:
            run_row = session.get(Run, run_id)
            if run_row is not None:
                for r in list(run_row.results):
                    session.delete(r)
                session.delete(run_row)
