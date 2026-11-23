"""Per-case trajectory metrics: what an agent *did*, not just what it answered
(docs/AGENTIC_PRD.md Phase 2). All pure functions of an `AgentRun` (plus the
case's `expected_tools`), so they can be computed from a stored trajectory
long after the run, with no model call.

A metric that is undefined for a case, such as a tool error rate when no tool
was called, is `None`, never 0: "no data" must not read as "perfect" or
"broken". Diff and gate already treat `None` as absent.
"""
from __future__ import annotations

from sagwa.adapters.base import AgentRun


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_trajectory_metrics(run: AgentRun, expected_tools: list[str] | None) -> dict:
    tools = [s for s in run.trajectory if s.step_type == "tool"]
    handoffs = [s for s in run.trajectory if s.step_type == "handoff"]
    failed = [i for i, s in enumerate(tools) if s.error]

    # Tool selection: the share of calls that went to a tool the case expects.
    accuracy = (
        _rate(sum(s.name in expected_tools for s in tools), len(tools))
        if expected_tools is not None else None
    )
    # A retry is a call straight after a failed call to the same tool.
    retries = sum(1 for i in range(1, len(tools)) if tools[i - 1].error and tools[i].name == tools[i - 1].name)
    # A loop is the third or later identical call (same tool, same input): one
    # retry after a failure is reasonable policy, a third attempt at the same
    # call is the agent going round in circles.
    seen: dict[tuple, int] = {}
    repeats = 0
    for s in tools:
        key = (s.name, s.input)
        seen[key] = seen.get(key, 0) + 1
        repeats += seen[key] >= 3
    # Recovery: a failed call later followed by a successful call to the same tool.
    recovered = sum(1 for i in failed if any(t.name == tools[i].name and not t.error for t in tools[i + 1:]))
    # A lossy handoff passes on less than the sender produced.
    lossy = sum(1 for s in handoffs if s.input is not None and s.output != s.input)

    return {
        "steps": run.step_count,
        "tool_calls": len(tools),
        "tool_call_accuracy": accuracy,
        "tool_error_rate": _rate(len(failed), len(tools)),
        "tool_retry_rate": _rate(retries, len(tools)),
        "loop_rate": _rate(repeats, len(tools)),
        "recovery_rate": _rate(recovered, len(failed)),
        "step_error": any(s.error for s in run.trajectory),
        "handoff_count": len(handoffs),
        "handoff_loss_rate": _rate(lossy, len(handoffs)),
        "tokens": run.tokens,
        "latency_ms": run.latency_ms,
        "cost_usd": run.cost_usd,
    }


def _per_success(results, field: str) -> float | None:
    """Σ field / number of successful cases, over a run's non-errored results.
    `None` when any of them lacks the field (a partial sum would understate
    it) or when nothing succeeded."""
    scored = [r for r in results if r.error is None]
    values = [getattr(r, field) for r in scored]
    successes = sum(1 for r in scored if ((r.metrics_json or {}).get("task") or {}).get("success") is True)
    if not scored or any(v is None for v in values) or not successes:
        return None
    return sum(values) / successes


def cost_per_success(results) -> float | None:
    return _per_success(results, "cost_usd")


def tokens_per_success(results) -> float | None:
    return _per_success(results, "tokens")
