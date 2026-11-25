"""`sagwa improve` end to end, offline: a fake agent whose behavior depends on
its prompt text, a fake LLM proposing revisions, the real run / diagnose /
diff / decide machinery in between (docs/AGENTIC_PRD.md Phase 6)."""
import pytest

from sagwa.adapters.base import AgentRun, AgentStep
from sagwa.datasets.schema import GoldenCase, SuccessCriteria
from sagwa.improve.graph import improve
from sagwa.prompts import PromptStore

CASES = [GoldenCase(id=f"c{i:02d}", input=f"question {i}", task_type="agentic", expected_tools=[],
                    success_criteria=SuccessCriteria(answer_matches=r"\bcorrect\b")) for i in range(12)]
GATES = {"task.success": {"op": "gte", "value": 0.7}, "agent.steps": {"op": "lte", "value": 12}}
PLANNER = "Plan the answer to: {question}"


def _agent(store_root, works_when):
    """An agent that answers correctly only when its planner prompt contains
    `works_when`: a failure a prompt revision can genuinely fix."""
    store = PromptStore(store_root)

    class FakeAgent:
        name = "fake-agent"
        prompt_store = store
        prompt_targets = {"GROUNDING": "planner"}

        def __init__(self, prompt_versions=None):
            self.prompt_versions = {"planner": (prompt_versions or {}).get("planner") or store.current("planner")}
            self.plan = store.load("planner", self.prompt_versions["planner"])

        def run(self, case_input):
            answer = "correct" if works_when in self.plan else "wrong"
            return AgentRun(answer=answer, context=None, latency_ms=1, tokens=10, cost_usd=None, trajectory=[
                AgentStep("agent", "planner", input=self.plan.format(question=case_input), output="plan"),
                AgentStep("agent", "writer", output=answer)])

    return FakeAgent, store


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # no judge calls
    global DATASET
    DATASET = tmp_path / "golden.jsonl"  # runs record the dataset's hash, so it must exist
    DATASET.write_text("".join(c.model_dump_json() + "\n" for c in CASES))


def _improve(agent_cls, llm, approve=lambda q: True, **kwargs):
    return improve(adapter_cls=agent_cls, target="tests:FakeAgent", dataset=DATASET, cases=CASES,
                   gates_config=GATES, llm_call=llm, approve=approve, **kwargs)


def _seeded(tmp_path, works_when="Verify"):
    agent, store = _agent(tmp_path / "prompts", works_when)
    store.add_version("planner", PLANNER)
    store.set_current("planner", "v1")
    return agent, store


def test_a_real_fix_is_accepted_and_becomes_current(tmp_path):
    agent, store = _seeded(tmp_path)
    prompts_seen = []
    fix = lambda prompt: prompts_seen.append(prompt) or PLANNER + " End with a Verify: step."
    state = _improve(agent, fix, max_iterations=1)
    assert "ACCEPT" in " ".join(state["log"]) and store.current("planner") == "v2"
    assert store.load("planner") == PLANNER + " End with a Verify: step."
    assert "Cause: GROUNDING" in prompts_seen[0] and "Q: question" in prompts_seen[0]
    assert state["outcome"] == "stopped after 1 iteration(s)"


def test_after_an_accepted_fix_the_next_iteration_finds_nothing_left(tmp_path):
    agent, store = _seeded(tmp_path)
    state = _improve(agent, lambda p: PLANNER + " Verify: it.", max_iterations=3)
    assert store.current("planner") == "v2" and state["outcome"] == "no failing cases: nothing to improve"


def test_a_revision_that_doesnt_help_is_rejected_and_changes_nothing(tmp_path):
    """The adversarial check: a loop that always 'improves' is measuring noise."""
    agent, store = _seeded(tmp_path)
    state = _improve(agent, lambda p: PLANNER + " Be thorough.", max_iterations=3)
    assert store.current("planner") == "v1" and store.versions("planner") == ["v1", "v2"]  # v2 kept as tried
    assert any("REJECT" in line and line.endswith(": no change") for line in state["log"])
    assert state["outcome"] == "no untried cause that a prompt revision can address"  # GROUNDING already tried


def test_a_baseline_with_no_failures_changes_nothing(tmp_path):
    agent, store = _seeded(tmp_path, works_when="Plan")  # the v1 planner already works
    calls = []
    state = _improve(agent, lambda p: calls.append(p) or "unused", max_iterations=3)
    assert state["outcome"] == "no failing cases: nothing to improve"
    assert calls == [] and store.versions("planner") == ["v1"] and store.current("planner") == "v1"


def test_an_invalid_revision_is_refused_before_anything_is_written(tmp_path):
    agent, store = _seeded(tmp_path)
    state = _improve(agent, lambda p: "Plan the answer.", max_iterations=1)  # drops {question}
    assert "drops placeholder(s) ['question']" in state["outcome"] and store.versions("planner") == ["v1"]


def test_dry_run_proposes_without_writing_or_running(tmp_path):
    agent, store = _seeded(tmp_path)
    state = _improve(agent, lambda p: PLANNER + " Verify: it.", dry_run=True)
    assert state["outcome"] == "dry run: proposed, not applied" and store.versions("planner") == ["v1"]
    assert "candidate_run_id" not in state


def test_an_accepted_fix_needs_approval_to_become_current(tmp_path):
    agent, store = _seeded(tmp_path)
    state = _improve(agent, lambda p: PLANNER + " Verify: it.", approve=lambda q: False)
    assert store.current("planner") == "v1" and "not approved" in state["outcome"]


def test_reusing_a_stored_baseline_run_skips_the_baseline_evaluation(tmp_path):
    from sagwa.cli import execute_run

    agent, store = _seeded(tmp_path)
    baseline, _ = execute_run(agent(), "tests:FakeAgent", DATASET, CASES)
    state = _improve(agent, lambda p: PLANNER + " Verify: it.", baseline_run_id=baseline, max_iterations=1)
    assert state["log"][0] == f"baseline: existing run {baseline}" and store.current("planner") == "v2"


def test_an_adapter_without_a_prompt_store_is_refused():
    class Plain:
        pass

    with pytest.raises(ValueError, match="prompt_store"):
        _improve(Plain, lambda p: "")


def test_cli_wires_options_through_and_prints_the_log(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sagwa.cli import app

    seen = {}

    def fake_improve(**kwargs):
        seen.update(kwargs)
        return {"log": ["diagnose: GROUNDING on 2 case(s)"], "outcome": "dry run: proposed, not applied"}

    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr("sagwa.improve.graph.improve", fake_improve)
    monkeypatch.setattr("sagwa.cli._load_adapter_class", lambda target: object)
    result = CliRunner().invoke(app, ["improve", "--target", "x:Y", "--dataset", str(DATASET),
                                      "--dry-run", "--max-iterations", "2", "--baseline-run", "r1"])
    assert result.exit_code == 0, result.output
    assert "diagnose: GROUNDING" in result.output and "dry run: proposed" in result.output
    assert seen["dry_run"] is True and seen["max_iterations"] == 2 and seen["baseline_run_id"] == "r1"
    asked = []
    monkeypatch.setattr("typer.confirm", lambda question, default: asked.append(question) or False)
    assert seen["approve"]("Make it current?") is False and asked == ["Make it current?"]  # no --yes: ask


def test_cli_needs_a_key_to_write_revisions(monkeypatch):
    from typer.testing import CliRunner

    from sagwa.cli import app

    result = CliRunner().invoke(app, ["improve", "--target", "x:Y", "--dataset", str(DATASET)])
    assert result.exit_code == 1 and "needs GROQ_API_KEY" in result.output
