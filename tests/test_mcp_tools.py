"""Sagwa's MCP tools (docs/AGENTIC_PRD.md Phase 7), called the way an agent
calls them: through an MCP client session against the real server object,
on a seeded database. Every answer must match what the CLI's function gives."""
import json

import anyio
import pytest
from mcp import Client

from sagwa.mcp.server import mcp

SIX_STEP = "tests.fixtures.fake_agent:SixStepAgent"


def call(name: str, **arguments):
    async def go():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def ok(name: str, **arguments) -> dict:
    result = call(name, **arguments)
    assert not result.is_error, result.content
    return result.structured_content


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def gates(tmp_path):
    path = tmp_path / "gates.yaml"
    path.write_text("metrics:\n  safety.pii.flagged: {op: lte, value: 0.0}\n"
                    "  reference.exact_match: {op: gte, value: 0.5}\nregression:\n  tolerance: 0.0\n")
    return str(path)


@pytest.fixture
def agentic_dataset(tmp_path):
    """One agentic case the six-step fixture agent fails: it answers Tokyo."""
    path = tmp_path / "agentic.jsonl"
    path.write_text(json.dumps({"id": "a-1", "input": "Capital of France?", "task_type": "agentic",
                                "expected_tools": ["search"],
                                "success_criteria": {"answer_matches": "Paris"}}) + "\n")
    return str(path)


def test_the_server_offers_the_eight_tools():
    async def go():
        async with Client(mcp) as client:
            return sorted(t.name for t in (await client.list_tools()).tools)

    assert anyio.run(go) == ["check_gate", "cluster_failures", "compare_runs", "get_run", "get_trace",
                             "inspect_failure", "root_causes", "run_evaluation"]


def test_did_my_change_regress_it_through_tool_calls_alone(gates):
    """The PRD's sequence: run -> compare -> gate -> inspect a failure."""
    baseline = ok("run_evaluation", target="stub", dataset="golden_sets/example.jsonl")
    candidate = ok("run_evaluation", target="stub", dataset="golden_sets/example.jsonl")
    assert baseline["cases"] == candidate["cases"] == 3 and baseline["errored"] == 0

    diff = ok("compare_runs", baseline_run_id=baseline["run_id"], candidate_run_id=candidate["run_id"],
              gates_config=gates)
    assert diff["candidate_run_id"] == candidate["run_id"] and diff["flips"] == []
    assert diff["table"].startswith("Diff: baseline=")

    gate = ok("check_gate", run_id=candidate["run_id"], gates_config=gates, baseline_run_id=baseline["run_id"])
    assert gate["passed"] is False  # the stub echoes, so exact match fails the absolute floor
    failing = next(m for m in gate["metric_results"] if not m["passed"])
    assert failing["metric_name"] == "reference.exact_match"

    case = ok("inspect_failure", run_id=candidate["run_id"], case_id="ex-001")
    assert case["output"].startswith("[stub echo]") and case["metrics_json"]["reference"]["exact_match"] == 0.0
    assert case["steps"] == 0 and case["trace_summary"] == []


def test_get_run_reports_metadata_and_counts():
    run = ok("run_evaluation", target="stub", dataset="golden_sets/example.jsonl")
    info = ok("get_run", run_id=run["run_id"])
    assert info["target"] == "stub" and info["cases"] == 3 and info["errored"] == 0 and info["status"] == "completed"
    assert info["task_success"] is None  # not an agentic run


def test_agent_tools_expose_the_trajectory_and_its_root_cause(agentic_dataset):
    run = ok("run_evaluation", target=SIX_STEP, dataset=agentic_dataset)
    assert ok("get_run", run_id=run["run_id"])["task_success"] == 0

    trace = ok("get_trace", run_id=run["run_id"], case_id="a-1")
    assert [s["seq"] for s in trace["steps"]] == list(range(6))
    assert trace["steps"][2]["error"] == "TimeoutError: search timed out"

    case = ok("inspect_failure", run_id=run["run_id"], case_id="a-1")
    assert case["metrics_json"]["task"]["success"] is False and case["steps"] == 6
    assert {"step_type": "tool", "name": "search", "steps": 2, "errors": 1} == {
        k: v for k, v in next(r for r in case["trace_summary"] if r["name"] == "search").items()
        if k in ("step_type", "name", "steps", "errors")}

    causes = ok("root_causes", run_id=run["run_id"])
    assert [c["case_ids"] for c in causes["causes"]] == [["a-1"]] and causes["report"].startswith("Root causes")


def test_cluster_failures_returns_clusters_as_data(gates):
    run = ok("run_evaluation", target="stub", dataset="golden_sets/example.jsonl")
    clusters = ok("cluster_failures", run_id=run["run_id"], gates_config=gates, min_cluster_size=2)
    from sagwa.clustering import failing_results
    from sagwa.gate import load_gate_config
    from sagwa.storage import get_session

    with get_session() as session:
        failing = {r.case_id for r in failing_results(session, run["run_id"], load_gate_config(gates))}
    assert clusters["run_id"] == run["run_id"] and failing  # same failing set the CLI clusters
    assert sorted(cid for c in clusters["clusters"] for cid in c["case_ids"]) == sorted(failing)


@pytest.mark.parametrize("tool, arguments, message", [
    ("get_run", {"run_id": "nope"}, "unknown run id: nope"),
    ("get_trace", {"run_id": "nope", "case_id": "x"}, "unknown run id: nope"),
    ("check_gate", {"run_id": "nope"}, "unknown run id: nope"),
    ("run_evaluation", {"target": "stub", "dataset": "missing.jsonl"}, "file not found"),
])
def test_expected_failures_reach_the_agent_with_their_message(tool, arguments, message):
    result = call(tool, **arguments)
    assert result.is_error and message in result.content[0].text


def test_a_missing_case_is_a_clear_error():
    run = ok("run_evaluation", target="stub", dataset="golden_sets/example.jsonl")
    result = call("get_trace", run_id=run["run_id"], case_id="nope")
    assert result.is_error and f"run {run['run_id']} has no case nope" in result.content[0].text
