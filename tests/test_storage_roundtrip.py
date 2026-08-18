from sagwa.storage import Result, Run, get_session


def test_insert_and_read_back_run_and_result():
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

        session.add(
            Result(
                run_id=run_id,
                case_id="ex-001",
                input="hello",
                output="[stub echo] hello",
                latency_ms=1,
            )
        )

    try:
        with get_session() as session:
            fetched = session.get(Run, run_id)
            assert fetched is not None
            assert fetched.target_name == "stub"
            assert len(fetched.results) == 1
            assert fetched.results[0].case_id == "ex-001"
    finally:
        with get_session() as session:
            fetched = session.get(Run, run_id)
            if fetched is not None:
                for result in list(fetched.results):
                    session.delete(result)
                session.delete(fetched)
