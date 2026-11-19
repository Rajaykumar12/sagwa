"""A fixture agentic adapter: returns an `AgentRun` with a fixed six-step
trajectory covering every step type, for tests/test_trace_storage.py."""
from sagwa.adapters.base import AgentRun, AgentStep

TRAJECTORY = [
    AgentStep("agent", "planner", input="plan", output="search, then answer",
              latency_ms=40, tokens=120, cost_usd=0.001, metadata={"model": "m-small", "prompt_version": "p1"}),
    AgentStep("tool", "search", input="capital of Japan", output="Tokyo is the capital",
              latency_ms=15, metadata={"arguments": {"query": "capital of Japan"}}),
    AgentStep("tool", "search", input="population of Tokyo", output=None,
              latency_ms=900, error="TimeoutError: search timed out", metadata={"arguments": {"query": "population"}}),
    AgentStep("handoff", "planner->writer", metadata={"reason": "enough context"}),
    AgentStep("agent", "writer", input="write answer", output="Tokyo",
              latency_ms=30, tokens=80, cost_usd=0.0005, metadata={"model": "m-large"}),
    AgentStep("tool", "finish", input="Tokyo", output="ok", latency_ms=1),
]


class SixStepAgent:
    name = "six-step-agent"

    def run(self, case_input: str) -> AgentRun:
        return AgentRun(answer="Tokyo", context=None, latency_ms=986, tokens=200,
                        cost_usd=0.0015, trajectory=list(TRAJECTORY))
