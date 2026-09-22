"""`sagwa run`'s persistence contract: results are written per case as they
complete, `--resume` finishes an interrupted run, and `--json` reports the run
id so CI needn't scrape stdout.

Three benchmark runs were lost in 2026-09-20/21 to a laptop suspend and an
exhausted token budget, because results were only written once every case had
finished. These pin the fix.
"""
import json

import pytest
from typer.testing import CliRunner

from sagwa.cli import app
from sagwa.storage import Result, Run, get_session

runner = CliRunner()
DATASET = "golden_sets/example.jsonl"  # 3 cases, ships with the repo


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Metrics are not what these tests are about, and computing them for real
    would mean live judge/RAGAS calls."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("sagwa.cli.compute_metrics", lambda case, answer, context: {"stub": {"ok": 1.0}})


def _run(*args):
    result = runner.invoke(app, ["run", "--target", "stub", "--dataset", DATASET, *args])
    assert result.exit_code == 0, result.output
    return result


def _results(run_id):
    with get_session() as session:
        return sorted(
            case_id for (case_id,) in session.query(Result.case_id).filter(Result.run_id == run_id)
        )


def test_run_writes_json_with_the_run_id(tmp_path):
    out = tmp_path / "run.json"
    _run("--json", str(out))

    payload = json.loads(out.read_text())
    assert payload["cases"] == 3
    assert payload["errored"] == 0
    assert payload["target"] == "stub"
    with get_session() as session:
        assert session.get(Run, payload["run_id"]) is not None


def test_results_are_persisted_during_the_run_not_after_it(monkeypatch, tmp_path):
    """The fix for three runs lost to suspends and rate limits: a case is
    committed as it finishes, so an interruption keeps what it already paid
    for. Asserted by observing the row count grow *while* the run is going."""
    counts_seen_mid_run = []

    def _counting_metrics(case, answer, context):
        with get_session() as session:
            counts_seen_mid_run.append(session.query(Result).count())
        return {"stub": {"ok": 1.0}}

    monkeypatch.setattr("sagwa.cli.compute_metrics", _counting_metrics)

    out = tmp_path / "run.json"
    _run("--concurrency", "1", "--json", str(out))
    run_id = json.loads(out.read_text())["run_id"]

    assert len(counts_seen_mid_run) == 3
    # Each case sees the previous one already committed — which is only true
    # if writes happen per case rather than in one batch at the end.
    assert counts_seen_mid_run[1] == counts_seen_mid_run[0] + 1
    assert counts_seen_mid_run[2] == counts_seen_mid_run[1] + 1
    assert len(_results(run_id)) == 3


def test_resume_reruns_only_the_missing_cases(tmp_path):
    out = tmp_path / "run.json"
    _run("--json", str(out))
    run_id = json.loads(out.read_text())["run_id"]
    all_case_ids = _results(run_id)

    # Simulate the interruption: one case never made it to the database.
    with get_session() as session:
        session.query(Result).filter(
            Result.run_id == run_id, Result.case_id == all_case_ids[0]
        ).delete()

    assert len(_results(run_id)) == 2

    resumed = tmp_path / "resumed.json"
    result = runner.invoke(
        app,
        ["run", "--target", "stub", "--dataset", DATASET, "--resume", run_id, "--json", str(resumed)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(resumed.read_text())

    assert payload["run_id"] == run_id, "resume must extend the run, not start a new one"
    assert payload["ran_now"] == 1, "only the missing case should be re-run"
    assert payload["cases"] == 3
    assert _results(run_id) == all_case_ids


def test_resume_of_a_complete_run_does_nothing(tmp_path):
    out = tmp_path / "run.json"
    _run("--json", str(out))
    run_id = json.loads(out.read_text())["run_id"]

    resumed = tmp_path / "resumed.json"
    result = runner.invoke(
        app,
        ["run", "--target", "stub", "--dataset", DATASET, "--resume", run_id, "--json", str(resumed)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(resumed.read_text())["ran_now"] == 0
    assert len(_results(run_id)) == 3


def test_resume_with_unknown_run_id_exits_nonzero():
    result = runner.invoke(
        app, ["run", "--target", "stub", "--dataset", DATASET, "--resume", "no-such-run"]
    )
    assert result.exit_code == 1
    assert "Unknown run id" in result.output


def test_run_marks_the_run_completed():
    out_run = _run()
    run_id = out_run.output.split()[1].rstrip(":")
    with get_session() as session:
        assert session.get(Run, run_id).status == "completed"


# --- sagwa migrate -------------------------------------------------------
# A consuming repo has no alembic.ini and no migrations/ directory, so
# `alembic upgrade head` — the command the local docs give — cannot work
# there. `sagwa migrate` is what the GitHub Action runs instead.


def test_migrate_creates_the_schema_in_an_empty_database(tmp_path, monkeypatch):
    db = tmp_path / "fresh.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")

    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output

    import sqlite3

    tables = {
        row[0] for row in sqlite3.connect(db).execute("select name from sqlite_master where type='table'")
    }
    assert {"runs", "results", "alembic_version"} <= tables


def test_migrate_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}")
    assert runner.invoke(app, ["migrate"]).exit_code == 0
    assert runner.invoke(app, ["migrate"]).exit_code == 0


def test_migrate_falls_back_to_the_packaged_migrations(tmp_path, monkeypatch):
    """Run from anywhere but the repo root — the case that matters, since an
    installed sagwa has no ./migrations to find."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'elsewhere.db'}")
    monkeypatch.chdir(tmp_path)

    from pathlib import Path as _Path

    import sagwa

    packaged = _Path(sagwa.__file__).parent / "_migrations"
    if not packaged.is_dir():
        pytest.skip("editable install: migrations are only copied into the package by a wheel build")

    assert runner.invoke(app, ["migrate"]).exit_code == 0
