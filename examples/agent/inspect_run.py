"""Scores stored agent runs with Sagwa's trajectory metrics, rebuilding each
case's `AgentRun` from its `results` row and `trace_steps`. No model call, so
a run can be (re)scored at any time for free.

Phase 1 needs each regression variant shown exhibiting its failure, and
Phase 2 needs the metrics shown separating the variants. Both read from here:

    python -m examples.agent.inspect_run <run_id> [<run_id> ...] [-v]
"""
import sys
from collections import defaultdict
from statistics import mean

from sagwa.adapters.base import AgentRun, AgentStep
from sagwa.datasets import load_golden_set
from sagwa.metrics.agent_success import check_success
from sagwa.metrics.agent_trajectory import compute_trajectory_metrics
from sagwa.storage import Result, Run, get_session

CASES = {c.id: c for c in load_golden_set("golden_sets/agentic_sandbox.jsonl")}
SHOWN = ["tool_call_accuracy", "tool_error_rate", "tool_retry_rate", "loop_rate",
         "recovery_rate", "handoff_loss_rate", "steps", "tokens"]


def agent_run(result: Result) -> AgentRun:
    return AgentRun(
        answer=result.output, context=result.context, latency_ms=result.latency_ms,
        tokens=result.tokens, cost_usd=result.cost_usd,
        trajectory=[AgentStep(s.step_type, s.name, input=s.input, output=s.output, latency_ms=s.latency_ms,
                              tokens=s.tokens, cost_usd=s.cost_usd, error=s.error, metadata=s.metadata_json or {})
                    for s in result.trace_steps],
    )


def score(run_id: str, verbose: bool) -> None:
    with get_session() as session:
        target = session.get(Run, run_id).target_name.split(":")[-1]
        results = sorted(session.query(Result).filter(Result.run_id == run_id).all(), key=lambda r: r.case_id)
        errored = [r for r in results if r.error]
        by_tag, values, rows = defaultdict(list), defaultdict(list), []
        for r in results:
            if r.error:
                continue
            case, run = CASES[r.case_id], agent_run(r)
            failed = check_success(case.success_criteria, run)
            m = compute_trajectory_metrics(run, case.expected_tools)
            by_tag[case.tags[0]].append(not failed)
            reviewer = any(s.name == "reviewer" for s in run.trajectory)
            for k in SHOWN:
                if m[k] is not None:
                    values[k].append(m[k])
            values["reviewer_ran"].append(reviewer)
            rows.append((r.case_id, case.tags[0], not failed, ",".join(failed), m["steps"],
                         "+".join(s.name for s in run.trajectory if s.step_type == "tool")[:50], (r.output or "")[:80]))

    done = sum(len(v) for v in by_tag.values())
    print(f"\n{target} ({run_id[:8]}): {len(results)} cases, {len(errored)} errored, "
          f"task.success {sum(sum(v) for v in by_tag.values())}/{done}")
    print("  by tag: " + ", ".join(f"{t} {sum(v)}/{len(v)}" for t, v in sorted(by_tag.items())))
    print("  " + " | ".join(f"{k} {mean(values[k]):.2f}" if values[k] else f"{k} n/a"
                            for k in SHOWN + ["reviewer_ran"]))
    if verbose:
        for row in rows:
            print(f"    {row[0]} {row[1][:9]:9s} {'OK ' if row[2] else 'BAD'} {row[3]:28s} {row[4]:2d} {row[5]:50s} {row[6]!r}")


if __name__ == "__main__":
    for run_id in [a for a in sys.argv[1:] if not a.startswith("-")]:
        score(run_id, "-v" in sys.argv)
