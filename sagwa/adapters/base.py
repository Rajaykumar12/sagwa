"""Target-pipeline adapter contract (PRD FR-3a).

Any target pipeline is integrated by implementing `TargetAdapter` — zero
required changes to the target's own codebase.

An agentic target implements `AgentAdapter` instead and returns an `AgentRun`:
the same final-answer fields plus the trajectory of steps that produced it
(PRD FR-29). `AgentRun` subclasses `AdapterResult`, so every
consumer of an `AdapterResult` accepts one unchanged, and existing adapters
need no edits.
"""
from dataclasses import dataclass, field, fields
from typing import Literal, Protocol

StepType = Literal["agent", "tool", "handoff"]


@dataclass
class AdapterResult:
    answer: str
    context: str | None
    latency_ms: int
    tokens: int | None
    cost_usd: float | None


@dataclass
class AgentStep:
    """One step of an agent's trajectory: an agent turn, a tool call, or a
    handoff between agents. Stored one row per step in `trace_steps`.

    Tool arguments, the model an agent turn used, its prompt version and the
    like go in `metadata`, so a new kind of detail needs no schema change."""

    step_type: StepType
    name: str
    input: str | None = None
    output: str | None = None
    latency_ms: int | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentRun(AdapterResult):
    """An `AdapterResult` plus how it was produced. The trajectory is the single
    source of truth for what the agent did; counts are derived from it, never
    stored beside it, so they can't disagree."""

    trajectory: list[AgentStep] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.trajectory)


def to_agent_run(result: AdapterResult) -> AgentRun:
    """Lifts a plain `AdapterResult` into an `AgentRun` with an empty
    trajectory, so the runner and storage handle one type downstream."""
    if isinstance(result, AgentRun):
        return result
    return AgentRun(**{f.name: getattr(result, f.name) for f in fields(AdapterResult)})


class TargetAdapter(Protocol):
    name: str

    def run(self, case_input: str) -> AdapterResult:
        ...


class AgentAdapter(Protocol):
    name: str

    def run(self, case_input: str) -> AgentRun:
        ...
