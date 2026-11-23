"""The Brindle Home agent's control flow, driven by a scripted fake model so it
runs offline: each regression variant must change exactly the behavior it
claims to, before any metric is asked to detect it."""
from dataclasses import replace
from types import SimpleNamespace as NS

from examples.agent import agents
from examples.agent.orchestrator import AgentConfig, Orchestrator
from examples.agent.tools import AMBIGUOUS_DESCRIPTIONS, GENERIC_NAMES, shipment_tracker


def _msg(content="", tool_calls=()):
    calls = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(tool_calls)]
    return NS(content=content, tool_calls=calls or None)


class ScriptedClient:
    """Answers each call from `script(messages, tools)`; records every prompt."""

    def __init__(self, script):
        self.script, self.calls = script, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, tools=None, **kwargs):
        self.calls.append((messages, tools))
        message = self.script(messages, tools)
        return NS(choices=[NS(message=message)], usage=NS(total_tokens=10))


def _role(messages, tools):
    if tools:
        return "researcher"
    text = messages[0]["content"]
    return "planner" if "numbered plan" in text else "reviewer" if "You are the reviewer" in text else "writer"


def _script(plan="1. Look it up\n2. Verify: check the number", research=None, review="APPROVE"):
    research = research or (lambda messages: _msg("Findings: 13 late orders."))

    def script(messages, tools):
        role = _role(messages, tools)
        if role == "planner":
            return _msg(plan)
        if role == "researcher":
            return research(messages)
        if role == "reviewer":
            return _msg(review)
        return _msg("There were 13 late orders.")
    return script


def _run(config=None, **script_kwargs):
    client = ScriptedClient(_script(**script_kwargs))
    return Orchestrator(config or AgentConfig(), client=client).run("How many orders shipped late?"), client


def test_baseline_trajectory_plans_researches_hands_off_writes_and_verifies():
    def research(messages):
        if messages[-1]["role"] == "tool":
            return _msg("Findings: 13 late orders.")
        return _msg(tool_calls=[("order_db", '{"sql": "SELECT count(*) FROM orders"}')])

    run, _ = _run(research=research)
    assert [(s.step_type, s.name) for s in run.trajectory] == [
        ("agent", "planner"), ("agent", "researcher"), ("tool", "order_db"), ("agent", "researcher"),
        ("handoff", "researcher->writer"), ("agent", "writer"), ("agent", "reviewer"),
    ]
    assert run.trajectory[2].output.startswith("count(*)") and run.trajectory[2].error is None
    assert run.answer == "There were 13 late orders."
    assert run.tokens == 10 * 5  # five model calls


def _always_failing_sql(messages):
    # The researcher keeps issuing a query against a column that doesn't exist.
    last = messages[-1]
    if last["role"] == "user" and "Stop calling it" in last["content"]:
        return _msg("Findings: the database has no carrier information.")
    return _msg(tool_calls=[("order_db", '{"sql": "SELECT carrier FROM orders"}')])


def test_retry_cap_stops_a_failing_tool_after_two_errors():
    run, _ = _run(research=_always_failing_sql)
    tool_steps = [s for s in run.trajectory if s.step_type == "tool"]
    assert len(tool_steps) == 2
    assert all("no such column" in s.error for s in tool_steps)
    assert run.step_count < 12


def test_v3_without_retry_cap_loops_until_the_step_limit():
    run, _ = _run(config=replace(AgentConfig(), retry_cap=None), research=_always_failing_sql)
    tool_steps = [s for s in run.trajectory if s.step_type == "tool"]
    assert run.step_count >= AgentConfig().max_steps
    assert len(tool_steps) >= 10
    handoff = next(s for s in run.trajectory if s.step_type == "handoff")
    assert "step limit" in handoff.input


def test_reviewer_runs_only_when_the_plan_asks_for_verification():
    with_verify, _ = _run(plan="1. Query\n2. Verify: recount")
    without, _ = _run(plan="1. Query\n2. Answer")
    assert "reviewer" in [s.name for s in with_verify.trajectory]
    assert "reviewer" not in [s.name for s in without.trajectory]


def test_v4_planner_prompt_no_longer_asks_for_verification():
    assert "Verify:" in agents.PLANNER_V1 and "Verify:" not in agents.PLANNER_THIN


def test_reviewer_revise_triggers_one_rewrite():
    run, _ = _run(review="REVISE: the count is unsupported")
    assert [s.name for s in run.trajectory].count("writer") == 2


def test_v5_handoff_truncates_findings_and_marks_it():
    long_findings = "Findings: " + "fact. " * 100
    run, client = _run(config=replace(AgentConfig(), handoff_chars=150),
                       research=lambda messages: _msg(long_findings))
    handoff = next(s for s in run.trajectory if s.step_type == "handoff")
    assert handoff.input == long_findings.strip() and len(handoff.output) == 150
    assert handoff.metadata["truncated"] is True
    writer_prompt = next(m[0]["content"] for m, t in client.calls if _role(m, t) == "writer")
    assert handoff.output in writer_prompt and long_findings.strip() not in writer_prompt


def test_v2_gives_both_lookup_tools_the_same_description():
    _, client = _run(config=replace(AgentConfig(), tool_descriptions=dict(AMBIGUOUS_DESCRIPTIONS)))
    tools = next(t for m, t in client.calls if t)
    descriptions = {t["function"]["name"]: t["function"]["description"] for t in tools}
    assert descriptions["corpus_search"].split(".")[0] == descriptions["order_db"].split(".")[0]


def test_shipment_tracker_is_always_down():
    import pytest

    with pytest.raises(ConnectionError):
        shipment_tracker("1050")


def _tracker_research(messages):
    last = messages[-1]
    if last["role"] == "user" and "Stop calling it" in last["content"]:
        return _msg("Findings: the tracking service is unavailable.")
    return _msg(tool_calls=[("shipment_tracker", '{"order_id": "1050"}')])


def test_baseline_gives_up_on_the_outage_and_v3_retries_until_the_step_limit():
    baseline, _ = _run(research=_tracker_research)
    v3, _ = _run(config=replace(AgentConfig(), retry_cap=None), research=_tracker_research)
    calls = lambda run: [s for s in run.trajectory if s.step_type == "tool"]
    assert len(calls(baseline)) == 2 and all("503" in s.error for s in calls(baseline))
    assert v3.step_count >= AgentConfig().max_steps > 20  # golden cases allow at most 20


def test_v2_model_sees_generic_names_but_the_trajectory_records_real_ones():
    def research(messages):
        if messages[-1]["role"] == "tool":
            return _msg("Findings: 13.")
        return _msg(tool_calls=[("lookup", '{"sql": "SELECT count(*) FROM orders"}')])

    config = replace(AgentConfig(), tool_descriptions=dict(AMBIGUOUS_DESCRIPTIONS), tool_names=dict(GENERIC_NAMES))
    run, client = _run(config=config, research=research)
    seen = {t["function"]["name"] for t in next(t for m, t in client.calls if t)}
    assert {"search", "lookup"} <= seen and "order_db" not in seen
    tool = next(s for s in run.trajectory if s.step_type == "tool")
    assert tool.name == "order_db" and tool.error is None
