# ADR 0008: LangGraph runs `sagwa improve` and nothing else

**Status**: Accepted (2026-11-25)

## Context

`sagwa improve` is a loop with state: evaluate a baseline, diagnose its failures, revise one prompt, evaluate the candidate, then accept or reject, and go round again. The agentic plan (now PRD §5.10) scoped LangGraph to this workflow and kept it out of the reference agent, which is a hand-rolled loop.

The question was whether LangGraph should also run Sagwa's core path (`run → metrics → diff → gate`), for consistency.

## Decision

**No. LangGraph is imported only by `sagwa/improve/graph.py`, and installed only with the `improve` extra.**

- **The core path isn't a graph.** `sagwa run` maps a fixed function over cases, and `diff`/`gate` are pure functions over stored rows. A framework adds a dependency and a second way to read the control flow, and buys nothing.
- **The improve loop is a graph.** It branches (accept / reject / nothing to fix / invalid revision / dry run), loops with a budget, and carries state between steps. That is what LangGraph is for, and writing it as explicit nodes and edges makes each branch testable on its own.
- **Every node calls existing code.** `execute_run` (the body of `sagwa run`), `root_causes`, `diff_runs` and the decision rule are the same functions the CLI uses, so `improve` can never apply a different standard from `sagwa diff`.
- **Services stay out of the graph state.** The adapter class, the LLM call and the approval prompt are bound in a closure, so the state is plain data (versions, run ids, the decision, a log) and the graph needs no checkpointer.

## Consequences

- The core install doesn't pull in LangGraph. `sagwa improve` without it fails at import, with the extra named in `pyproject.toml`.
- If the core path ever needs branching orchestration (it doesn't today), revisit this; until then, a framework there would be speculative.
- The change surface stays at prompt text (locked decision #7): `improve` writes new prompt-store versions and moves a `current` pointer, and never writes or runs code.
