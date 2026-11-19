"""Sagwa adapters for the Brindle Home agent: a baseline and four variants,
each regressing exactly one thing (docs/AGENTIC_PRD.md Phase 1). Each is a
plausible PR, not sabotage: shorter tool docs, a removed safety limit, a
leaner prompt, a smaller payload.

    sagwa run --target examples.agent.adapter:BrindleAgentV1 --dataset golden_sets/agentic_sandbox.jsonl
"""
from dataclasses import replace

from examples.agent import agents
from examples.agent.orchestrator import AgentConfig, Orchestrator
from examples.agent.tools import AMBIGUOUS_DESCRIPTIONS
from sagwa.adapters.base import AgentRun


class BrindleAgentV1:
    name = "brindle-agent-v1"
    config = AgentConfig()

    def __init__(self):
        self.orchestrator = Orchestrator(self.config)
        self.model_label = agents.MODEL

    def run(self, case_input: str) -> AgentRun:
        return self.orchestrator.run(case_input)


class V2_ToolOverlap(BrindleAgentV1):
    """Tool descriptions trimmed until corpus_search and order_db read the same
    ("Look up information about Brindle Home"): the agent can't tell which
    tool holds policies and which holds orders."""

    name = "brindle-agent-v2-tool-overlap"
    config = replace(AgentConfig(), tool_descriptions=dict(AMBIGUOUS_DESCRIPTIONS))


class V3_NoRetryCap(BrindleAgentV1):
    """The consecutive-failure cap removed: a tool that keeps failing is
    retried until the hard step limit ends the case."""

    name = "brindle-agent-v3-no-retry-cap"
    config = replace(AgentConfig(), retry_cap=None)


class V4_ThinPlanner(BrindleAgentV1):
    """A leaner planner prompt that no longer asks for a verification step,
    so the reviewer never runs."""

    name = "brindle-agent-v4-thin-planner"
    config = replace(AgentConfig(), planner_prompt=agents.PLANNER_THIN)


class V5_LossyHandoff(BrindleAgentV1):
    """The researcher's findings capped at 150 characters on handoff, to keep
    the writer's prompt small: the writer gets the first fact, not all of them."""

    name = "brindle-agent-v5-lossy-handoff"
    config = replace(AgentConfig(), handoff_chars=150)
