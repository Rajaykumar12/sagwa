"""Sagwa adapters for the Brindle Home agent: a baseline and four variants,
each regressing exactly one thing (docs/AGENTIC_PRD.md Phase 1). Each is a
plausible PR, not sabotage: shorter tool docs, a removed safety limit, a
leaner prompt, a smaller payload.

    sagwa run --target examples.agent.adapter:BrindleAgentV1 --dataset golden_sets/agentic_sandbox.jsonl
"""
from dataclasses import replace

from examples.agent import agents
from examples.agent.orchestrator import AgentConfig, Orchestrator
from examples.agent.tools import GENERIC_NAMES
from sagwa.adapters.base import AgentRun


class BrindleAgentV1:
    name = "brindle-agent-v1"
    config = AgentConfig()
    # What `sagwa improve` may revise for each root cause: prompt text only.
    # HANDOFF has no entry, because V5's truncation is code, not a prompt.
    prompt_store = agents.STORE
    prompt_targets = {"TOOL_SELECTION": "tool_descriptions", "PLANNING": "planner", "SCOPE": "planner",
                      "BUDGET_EXHAUSTED": "planner", "INFINITE_RETRY": "researcher", "GROUNDING": "writer"}

    def __init__(self, prompt_versions: dict[str, str] | None = None):
        """`prompt_versions` overrides this variant's own pins; `sagwa improve`
        passes the candidate versions here."""
        config = replace(self.config, prompt_versions={**self.config.prompt_versions, **(prompt_versions or {})})
        self.orchestrator = Orchestrator(config)
        self.prompt_versions = self.orchestrator.prompt_versions  # recorded as Run.prompt_version
        self.model_label = agents.MODEL

    def run(self, case_input: str) -> AgentRun:
        return self.orchestrator.run(case_input)


class V2_ToolOverlap(BrindleAgentV1):
    """An "API cleanup" PR: corpus_search and order_db renamed to `search` and
    `lookup` and given the same description ("Look up information about
    Brindle Home"), so nothing tells the policy source from the order source.

    Vague descriptions alone did not regress anything (2026-11-23: gpt-oss-20b
    still picked the right tool every time, from the self-describing names),
    which is why the names change too."""

    name = "brindle-agent-v2-tool-overlap"
    config = replace(AgentConfig(), prompt_versions={"tool_descriptions": "v2"}, tool_names=dict(GENERIC_NAMES))


class V3_NoRetryCap(BrindleAgentV1):
    """The consecutive-failure cap removed: a tool that keeps failing (the
    shipment tracker is down) is retried until the hard step limit ends the
    case."""

    name = "brindle-agent-v3-no-retry-cap"
    config = replace(AgentConfig(), retry_cap=None)


class V4_ThinPlanner(BrindleAgentV1):
    """A leaner planner prompt that no longer asks for a verification step,
    so the reviewer never runs."""

    name = "brindle-agent-v4-thin-planner"
    config = replace(AgentConfig(), prompt_versions={"planner": "v2"})


class V5_LossyHandoff(BrindleAgentV1):
    """The researcher's findings capped at 150 characters on handoff, to keep
    the writer's prompt small: the writer gets the first fact, not all of them."""

    name = "brindle-agent-v5-lossy-handoff"
    config = replace(AgentConfig(), handoff_chars=150)
