"""Root-cause engine (docs/AGENTIC_PRD.md Phase 5) on fixture trajectories
whose cause is known: each category must come out exactly where it should."""
import json

import pytest
from typer.testing import CliRunner

from sagwa.adapters.base import AgentRun, AgentStep
from sagwa.cli import app
from sagwa.clustering import trajectory_signature
from sagwa.datasets.schema import GoldenCase, SuccessCriteria
from sagwa.rootcause import diagnose_case, group_diagnoses
from sagwa.storage import Result, Run, TraceStep, get_session


def _case(**criteria):
    expected = criteria.pop("expected_tools", ["order_db"])
    return GoldenCase(id="c1", input="q", task_type="agentic", expected_tools=expected,
                      success_criteria=SuccessCriteria(**{"answer_matches": r"\b13\b", **criteria}))


def _run(steps, answer="There were 13."):
    return AgentRun(answer=answer, context=None, latency_ms=1, tokens=1, cost_usd=None, trajectory=steps)


def _healthy(answer="There were 13.", reviewer=True):
    steps = [AgentStep("agent", "planner"), AgentStep("agent", "researcher"),
             AgentStep("tool", "order_db", input="SELECT 1", output="13"),
             AgentStep("handoff", "researcher->writer", input="13 late", output="13 late"),
             AgentStep("agent", "writer", output=answer)]
    if reviewer:
        steps.append(AgentStep("agent", "reviewer", output="APPROVE"))
    return _run(steps, answer)


def test_a_successful_case_has_no_diagnosis():
    assert diagnose_case(_case(), _healthy()) is None


def test_infinite_retry():
    steps = [AgentStep("agent", "planner")] + [
        AgentStep("tool", "shipment_tracker", input='{"order_id": "1050"}', error="ConnectionError: 503")
        for _ in range(10)] + [AgentStep("agent", "writer")]
    d = diagnose_case(_case(max_steps=8, must_not_error=False), _run(steps, "unknown"))
    assert d.category == "INFINITE_RETRY"
    assert "BUDGET_EXHAUSTED" in d.signals  # also true, but the loop explains it
    assert "`shipment_tracker` called 10× with identical input, 10 of them failed" in d.evidence


def test_budget_exhausted_without_a_loop():
    steps = [AgentStep("tool", "order_db", input=f"q{i}", output="x") for i in range(12)]
    d = diagnose_case(_case(max_steps=8), _run(steps))
    assert d.category == "BUDGET_EXHAUSTED" and d.signals == ["BUDGET_EXHAUSTED"] and d.confidence == 1.0


def test_handoff():
    run = _healthy(answer="I couldn't find it.")
    run.trajectory[3] = AgentStep("handoff", "researcher->writer", input="13 late orders, 2 in West", output="13 la")
    d = diagnose_case(_case(), run)
    assert d.category == "HANDOFF"
    assert "passed on 5 of 25 characters (20%)" in d.evidence[0]


def test_tool_selection_from_a_missed_required_tool():
    run = _healthy(answer="Probably 12.")
    run.trajectory[2] = AgentStep("tool", "corpus_search", input="late orders", output="policy text")
    d = diagnose_case(_case(required_tools=["order_db"]), run)
    assert d.category == "TOOL_SELECTION"
    assert d.evidence == ["used corpus_search; expected order_db (accuracy 0.00)"]


def test_planning_needs_a_baseline_that_ran_the_step():
    case = _case()
    thin = _healthy(answer="Probably 12.", reviewer=False)
    assert diagnose_case(case, thin).category == "GROUNDING"  # no baseline: no contrast
    d = diagnose_case(case, thin, baseline=_healthy())
    assert d.category == "PLANNING" and d.pre_existing is False
    assert d.evidence == ["step(s) reviewer ran in the baseline and were skipped here"]


def test_a_case_that_failed_in_the_baseline_too_isnt_blamed_on_the_difference():
    """Seen live (2026-11-23): V4's only failure had failed identically in the
    baseline, so the missing reviewer can't be what broke it."""
    case = _case()
    d = diagnose_case(case, _healthy(answer="Probably 12.", reviewer=False),
                      baseline=_healthy(answer="Probably 11."))
    assert d.pre_existing is True and d.category == "GROUNDING"


def test_grounding_is_reached_by_elimination_with_lower_confidence():
    d = diagnose_case(_case(), _healthy(answer="There were 12."))
    assert d.category == "GROUNDING" and d.confidence == 0.5


def test_unknown_when_nothing_explains_it():
    run = _healthy(answer="There were 12.")
    run.trajectory[2] = AgentStep("tool", "order_db", input="SELECT 1", error="OperationalError: locked")
    d = diagnose_case(_case(), run)  # failed answer + step error: not clean grounding, no other signal
    assert d.category == "UNKNOWN" and d.confidence == 0.0


def test_grouping_largest_cause_first_with_evidence_counts():
    handoff = _healthy(answer="?")
    handoff.trajectory[3] = AgentStep("handoff", "researcher->writer", input="abcdefghij", output="abc")
    diagnoses = [diagnose_case(_case().model_copy(update={"id": cid}), run) for cid, run in [
        ("h1", handoff), ("h2", handoff), ("g1", _healthy(answer="12"))]]
    causes = group_diagnoses(diagnoses)
    assert [(c.category, c.case_ids) for c in causes] == [("HANDOFF", ["h1", "h2"]), ("GROUNDING", ["g1"])]
    assert causes[0].evidence == ["`researcher->writer` passed on 3 of 10 characters (30%) (2 cases)"]


# --- storage-backed: trajectory signatures and the CLI ----------------------


@pytest.fixture
def stored_runs(tmp_path):
    dataset = tmp_path / "set.jsonl"
    dataset.write_text(json.dumps({"id": "c1", "input": "q", "task_type": "agentic", "expected_tools": ["order_db"],
                                   "success_criteria": {"answer_matches": r"\b13\b"}}) + "\n")
    ids = {}
    with get_session() as session:
        for name, answer, keep in [("baseline", "There were 13.", "13 late orders"), ("candidate", "?", "13")]:
            run = Run(sagwa_git_sha="t", target_name=name, model="m", dataset_path=str(dataset), dataset_sha256="x")
            session.add(run)
            session.flush()
            result = Result(run_id=run.id, case_id="c1", input="q", output=answer, latency_ms=1)
            session.add(result)
            session.flush()
            session.add_all([
                TraceStep(result_id=result.id, seq=0, step_type="agent", name="planner", metadata_json={}),
                TraceStep(result_id=result.id, seq=1, step_type="tool", name="order_db", input="a", output="x",
                          metadata_json={}),
                # Only the candidate's second query fails; the baseline is clean.
                TraceStep(result_id=result.id, seq=2, step_type="tool", name="order_db", input="b",
                          error=None if name == "baseline" else "OperationalError: no such column",
                          output="y" if name == "baseline" else None, metadata_json={}),
                TraceStep(result_id=result.id, seq=3, step_type="handoff", name="r->w", input="13 late orders",
                          output=keep, metadata_json={}),
            ])
            ids[name] = run.id
    return ids


def test_trajectory_signature_collapses_repeats_and_marks_errors_and_loss(stored_runs):
    with get_session() as session:
        result = session.query(Result).filter(Result.run_id == stored_runs["candidate"]).one()
        sig = trajectory_signature(result)
    assert sig == "agent planner -> tool order_db x2 [OperationalError] -> handoff r->w [lossy]"


def test_rootcause_cli_reports_the_cause_and_writes_json(stored_runs, tmp_path):
    out = tmp_path / "rc.json"
    result = CliRunner().invoke(app, ["rootcause", "--run-id", stored_runs["candidate"],
                                      "--baseline", stored_runs["baseline"], "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "HANDOFF — 1 case(s)" in result.output
    (cause,) = json.loads(out.read_text())
    assert cause["category"] == "HANDOFF" and cause["cases"][0]["pre_existing"] is False


def test_rootcause_cli_rejects_an_unknown_run():
    result = CliRunner().invoke(app, ["rootcause", "--run-id", "nope"])
    assert result.exit_code == 1 and "Unknown run id" in result.output
