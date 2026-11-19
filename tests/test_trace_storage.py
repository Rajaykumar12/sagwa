"""Agent trajectories are stored one row per step in `trace_steps` and come back
intact, in order (docs/AGENTIC_PRD.md Phase 0). Plain adapters keep working
exactly as before and store no steps."""
import json

import pytest
from typer.testing import CliRunner

from sagwa.adapters.base import AdapterResult, AgentRun, to_agent_run
from sagwa.cli import app
from sagwa.storage import Result, get_session
from tests.fixtures.fake_agent import TRAJECTORY

runner = CliRunner()
DATASET = "golden_sets/example.jsonl"  # 3 cases


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("sagwa.cli.compute_metrics", lambda case, answer, context: {})


def _run(target, tmp_path) -> str:
    out = tmp_path / "run.json"
    result = runner.invoke(app, ["run", "--target", target, "--dataset", DATASET, "--json", str(out)])
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text())["run_id"]


def test_six_step_trajectory_round_trips_through_sagwa_run(tmp_path):
    run_id = _run("tests.fixtures.fake_agent:SixStepAgent", tmp_path)

    with get_session() as session:
        results = session.query(Result).filter(Result.run_id == run_id).all()
        assert len(results) == 3
        for result in results:
            steps = result.trace_steps  # ordered by seq
            assert [s.seq for s in steps] == list(range(6))
            for stored, original in zip(steps, TRAJECTORY):
                assert (stored.step_type, stored.name, stored.input, stored.output) == (
                    original.step_type, original.name, original.input, original.output
                )
                assert (stored.latency_ms, stored.tokens, stored.cost_usd, stored.error) == (
                    original.latency_ms, original.tokens, original.cost_usd, original.error
                )
                assert stored.metadata_json == original.metadata
            # The case-level roll-up is unchanged by having a trajectory.
            assert result.output == "Tokyo" and result.cost_usd == 0.0015


def test_plain_adapter_stores_no_trace_steps(tmp_path):
    run_id = _run("stub", tmp_path)
    with get_session() as session:
        results = session.query(Result).filter(Result.run_id == run_id).all()
        assert len(results) == 3
        assert all(r.trace_steps == [] for r in results)
        assert all(r.trace_id is None for r in results)  # reserved for external tracing (ADR-0002)


def test_to_agent_run_lifts_an_adapter_result_with_an_empty_trajectory():
    plain = AdapterResult(answer="a", context="c", latency_ms=5, tokens=7, cost_usd=0.1)
    lifted = to_agent_run(plain)
    assert isinstance(lifted, AgentRun) and isinstance(lifted, AdapterResult)
    assert (lifted.answer, lifted.context, lifted.latency_ms, lifted.tokens, lifted.cost_usd) == ("a", "c", 5, 7, 0.1)
    assert lifted.trajectory == [] and lifted.step_count == 0


def test_to_agent_run_passes_an_agent_run_through_unchanged():
    run = AgentRun(answer="a", context=None, latency_ms=1, tokens=None, cost_usd=None, trajectory=list(TRAJECTORY))
    assert to_agent_run(run) is run
    assert run.step_count == 6
