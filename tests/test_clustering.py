import pytest

pytest.importorskip("hdbscan")
pytest.importorskip("sentence_transformers")

from sagwa.clustering import cluster_failures, cluster_run, failing_results, label_cluster
from sagwa.storage import Result, Run, get_session


def _make_result(run_id, case_id, input_text, output_text, exact_match):
    return Result(
        run_id=run_id,
        case_id=case_id,
        input=input_text,
        output=output_text,
        latency_ms=1,
        metrics_json={"reference": {"exact_match": exact_match}},
    )


def test_cluster_failures_rejects_min_cluster_size_below_two():
    results = [Result(run_id="r", case_id="1", input="x", output="y", latency_ms=1)]
    with pytest.raises(ValueError, match="min_cluster_size must be >= 2"):
        cluster_failures(results, min_cluster_size=1)


def test_label_cluster_keyword_fallback_without_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    label = label_cluster(["billing error: invoice not found", "billing error: payment declined"])
    assert isinstance(label, str)
    assert label != ""


def test_cluster_failures_separates_two_groups():
    billing = [
        Result(run_id="r", case_id=f"billing-{i}", input="billing question",
               output="billing error invoice payment declined refund", latency_ms=1)
        for i in range(4)
    ]
    login = [
        Result(run_id="r", case_id=f"login-{i}", input="login question",
               output="login error password reset authentication failed", latency_ms=1)
        for i in range(4)
    ]
    clusters = cluster_failures(billing + login, min_cluster_size=3)

    all_case_ids = {cid for c in clusters for cid in c.case_ids}
    assert all_case_ids == {r.case_id for r in billing + login}
    # every case landed in exactly one cluster (including possibly -1 noise)
    assert sum(len(c.case_ids) for c in clusters) == len(billing) + len(login)


def test_cluster_run_end_to_end_sorted_by_size(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # force keyword-fallback labeling, no network

    with get_session() as session:
        run_row = Run(
            sagwa_git_sha="a", target_name="stub", model="n/a",
            dataset_path="golden_sets/example.jsonl", dataset_sha256="0" * 64,
        )
        session.add(run_row)
        session.flush()
        run_id = run_row.id

        # 4 failing "billing" cases, 1 failing "login" case, 1 passing case
        for i in range(4):
            session.add(_make_result(run_id, f"billing-{i}", "billing question",
                                       "billing error invoice payment declined refund", exact_match=0.0))
        session.add(_make_result(run_id, "login-0", "login question",
                                   "login error password reset authentication failed", exact_match=0.0))
        session.add(_make_result(run_id, "passing-0", "ok question", "ok answer", exact_match=1.0))

    try:
        gates = {"reference.exact_match": {"op": "gte", "value": 1.0}}
        with get_session() as session:
            failing = failing_results(session, run_id, gates)
            assert {r.case_id for r in failing} == {f"billing-{i}" for i in range(4)} | {"login-0"}

            clusters = cluster_run(session, run_id, gates_config=gates, min_cluster_size=3)

        sizes = [c.size for c in clusters]
        assert sizes == sorted(sizes, reverse=True)
        assert all(c.label for c in clusters)
    finally:
        with get_session() as session:
            run_row = session.get(Run, run_id)
            if run_row is not None:
                for r in list(run_row.results):
                    session.delete(r)
                session.delete(run_row)
