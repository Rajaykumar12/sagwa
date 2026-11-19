"""Manual check of one agent run: reads its stored results and trace_steps
and evaluates each case's success_criteria, plus step, tool and handoff counts.

Phase 1 requires each regression variant to be shown exhibiting its failure
by hand before any metric claims to detect it; this is that check. It is not
a Sagwa metric: Phase 2's `sagwa/metrics/agent_success.py` replaces it.

    python -m examples.agent.inspect_run <run_id> [-v]
"""
import re
import sys
from collections import Counter, defaultdict

from sagwa.datasets import load_golden_set
from sagwa.storage import Result, get_session

run_id = sys.argv[1]
cases = {c.id: c for c in load_golden_set("golden_sets/agentic_sandbox.jsonl")}


def success(case, result, steps):
    sc, ans = case.success_criteria, result.output or ""
    tools = [s.name for s in steps if s.step_type == "tool"]
    why = []
    if result.error: why.append("case error")
    if any(x.lower() not in ans.lower() for x in sc.answer_contains): why.append("contains")
    if sc.answer_matches and not re.search(sc.answer_matches, ans, re.I): why.append("matches")
    if any(t not in tools for t in sc.required_tools): why.append("required_tools")
    if any(t in tools for t in sc.forbidden_tools): why.append("forbidden_tool")
    if sc.max_steps and len(steps) > sc.max_steps: why.append(f"steps>{sc.max_steps}")
    if sc.must_not_error and any(s.error for s in steps): why.append("step_error")
    return not why, why


by_tag = defaultdict(lambda: [0, 0])
stats = Counter()
rows = []
with get_session() as s:
    results = s.query(Result).filter(Result.run_id == run_id).all()
    for r in sorted(results, key=lambda r: r.case_id):
        case, steps = cases[r.case_id], list(r.trace_steps)
        ok, why = success(case, r, steps)
        tag = case.tags[0]
        by_tag[tag][0] += ok; by_tag[tag][1] += 1
        tools = [x.name for x in steps if x.step_type == "tool"]
        stats["steps"] += len(steps); stats["tool_calls"] += len(tools)
        stats["tool_errors"] += sum(1 for x in steps if x.step_type == "tool" and x.error)
        stats["reviewer"] += any(x.name == "reviewer" for x in steps)
        stats["truncated"] += any(x.step_type == "handoff" and x.metadata_json.get("truncated") for x in steps)
        stats["tokens"] += r.tokens or 0
        stats["wrong_tool_first"] += bool(tools) and bool(case.expected_tools) and tools[0] not in case.expected_tools
        rows.append((r.case_id, tag, ok, ",".join(why), len(steps), "+".join(tools)[:60], (r.output or r.error or "")[:90]))
n = len(results)
print(f"run {run_id}: {n} cases, success {sum(v[0] for v in by_tag.values())}/{n}")
print("by tag:", {t: f"{a}/{b}" for t, (a, b) in sorted(by_tag.items())})
print(f"mean steps {stats['steps']/n:.1f} | tool calls {stats['tool_calls']} | tool errors {stats['tool_errors']} | "
      f"reviewer ran {stats['reviewer']}/{n} | truncated handoffs {stats['truncated']}/{n} | "
      f"first tool not expected {stats['wrong_tool_first']} | tokens {stats['tokens']} ({stats['tokens']/n:.0f}/case)")
if "-v" in sys.argv:
    for row in rows:
        print(" ", row[0], row[1][:8], "OK " if row[2] else "BAD", f"{row[3]:24s}", row[4], row[5], "|", repr(row[6]))
