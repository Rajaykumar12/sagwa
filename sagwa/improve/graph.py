"""`sagwa improve`: evaluate → diagnose → revise a prompt → re-evaluate →
accept only if statistically better (docs/AGENTIC_PRD.md Phase 6).

LangGraph orchestrates this one stateful workflow and nothing else: it is
deliberately absent from Sagwa's core (ADR-0008). Every step reuses the
existing machinery: `execute_run` (the body of `sagwa run`), `root_causes`,
`diff_runs` and the decision rule in `decision.py`.

What it may change (locked decision #7): **prompt text only**, as a new
version in the target's prompt store. `current` moves only for a revision
that was accepted by the statistics *and* approved by a person (or
`--yes`). Rejected revisions stay in the store as a record of what was tried.

The target adapter must expose:
- `prompt_store`: a `sagwa.prompts.PromptStore`
- `prompt_targets`: root-cause category -> prompt name it may revise
- `__init__(prompt_versions=...)`, and a `prompt_versions` attribute
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class ImproveState(TypedDict, total=False):
    versions: dict[str, str]            # prompt versions of the current baseline
    baseline_run_id: str
    tried: list[str]                    # root-cause categories already attempted
    iteration: int
    cause: dict                         # the RootCause being addressed (to_dict)
    prompt_name: str
    candidate_version: str
    candidate_run_id: str
    decision: dict
    outcome: str                        # why the loop stopped
    log: list[str]


def build_graph(*, adapter_cls, target: str, dataset: Path, cases, gates_config: dict, llm_call: Callable[[str], str],
                approve: Callable[[str], bool], max_iterations: int = 3, dry_run: bool = False,
                primary: str = "task.success", concurrency: int = 2):
    from sagwa.cli import execute_run
    from sagwa.diff import diff_runs
    from sagwa.improve.decision import decide
    from sagwa.improve.intervention import propose_revision
    from sagwa.rootcause import root_causes
    from sagwa.storage import Result, get_session

    store = adapter_cls.prompt_store
    golden = {c.id: c for c in cases}

    def note(state, line):
        return state.get("log", []) + [line]

    def run_baseline(state: ImproveState) -> ImproveState:
        if state.get("baseline_run_id"):
            return {"log": note(state, f"baseline: existing run {state['baseline_run_id']}")}
        adapter = adapter_cls(prompt_versions=state.get("versions"))
        run_id, _ = execute_run(adapter, target, dataset, cases, concurrency=concurrency)
        return {"baseline_run_id": run_id, "versions": dict(adapter.prompt_versions),
                "log": note(state, f"baseline: ran {run_id} with {adapter.prompt_versions}")}

    def diagnose(state: ImproveState) -> ImproveState:
        with get_session() as session:
            causes = root_causes(session, state["baseline_run_id"], golden_cases=golden)
        fixable = [c for c in causes if c.category in adapter_cls.prompt_targets and c.category not in state.get("tried", [])]
        if not causes:
            return {"outcome": "no failing cases: nothing to improve", "log": note(state, "diagnose: no failures")}
        if not fixable:
            return {"outcome": "no untried cause that a prompt revision can address",
                    "log": note(state, "diagnose: " + ", ".join(f"{c.category}×{len(c.case_ids)}" for c in causes)
                                + " (none fixable by prompt text, or all tried)")}
        cause = fixable[0]
        return {"cause": cause.to_dict(), "prompt_name": adapter_cls.prompt_targets[cause.category],
                "tried": state.get("tried", []) + [cause.category],
                "log": note(state, f"diagnose: {cause.category} on {len(cause.case_ids)} case(s) → revise "
                                   f"`{adapter_cls.prompt_targets[cause.category]}`")}

    def intervene(state: ImproveState) -> ImproveState:
        from sagwa.rootcause import RootCause

        cause = RootCause(**{k: v for k, v in state["cause"].items() if k != "cases"})
        name, version = state["prompt_name"], state["versions"][state["prompt_name"]]
        with get_session() as session:
            rows = session.query(Result).filter(Result.run_id == state["baseline_run_id"],
                                                Result.case_id.in_(cause.case_ids)).all()
            examples = [f"Q: {r.input} | A: {r.output}" for r in rows]
        revised, problems = propose_revision(llm_call, name, store.load(name, version), cause, examples)
        if revised is None:
            return {"outcome": f"no valid revision for `{name}`: {'; '.join(problems)}",
                    "log": note(state, f"intervene: revision refused ({'; '.join(problems)})")}
        if dry_run:
            return {"outcome": "dry run: proposed, not applied",
                    "log": note(state, f"intervene (dry run): proposed `{name}` revision:\n{revised}")}
        new_version = store.add_version(name, revised, parent=version, note=f"sagwa improve: {cause.category}")
        return {"candidate_version": new_version, "log": note(state, f"intervene: wrote `{name}` {new_version}")}

    def run_candidate(state: ImproveState) -> ImproveState:
        versions = {**state["versions"], state["prompt_name"]: state["candidate_version"]}
        run_id, _ = execute_run(adapter_cls(prompt_versions=versions), target, dataset, cases, concurrency=concurrency)
        return {"candidate_run_id": run_id, "log": note(state, f"candidate: ran {run_id}")}

    def compare(state: ImproveState) -> ImproveState:
        with get_session() as session:
            diff = diff_runs(session, state["baseline_run_id"], state["candidate_run_id"],
                             gates_config=gates_config, golden_cases=golden)
        decision = decide(diff, gates_config, primary=primary)
        return {"decision": decision.to_dict(),
                "log": note(state, f"compare: {'ACCEPT' if decision.accepted else 'REJECT'} — {decision.reasons[0]}")}

    def accept(state: ImproveState) -> ImproveState:
        name, version = state["prompt_name"], state["candidate_version"]
        if not approve(f"Make `{name}` {version} the current version? {state['decision']['reasons'][0]}"):
            return {"outcome": "improvement found but not approved; current versions unchanged",
                    "log": note(state, f"accept: declined by the approver; `{name}` stays at {state['versions'][name]}")}
        store.set_current(name, version)
        return {"versions": {**state["versions"], name: version}, "baseline_run_id": state["candidate_run_id"],
                "iteration": state.get("iteration", 0) + 1,
                "log": note(state, f"accept: `{name}` current → {version}")}

    def reject(state: ImproveState) -> ImproveState:
        return {"iteration": state.get("iteration", 0) + 1,
                "log": note(state, f"reject: `{state['prompt_name']}` stays at {state['versions'][state['prompt_name']]}"
                                   f" ({state['candidate_version']} kept in the store as tried)")}

    def next_or_stop(state: ImproveState) -> str:
        if state.get("outcome"):
            return END
        if state.get("iteration", 0) >= max_iterations:
            return "stop_at_limit"
        return "diagnose"

    def stop_at_limit(state: ImproveState) -> ImproveState:
        return {"outcome": f"stopped after {max_iterations} iteration(s)"}

    graph = StateGraph(ImproveState)
    for name, fn in [("run_baseline", run_baseline), ("diagnose", diagnose), ("intervene", intervene),
                     ("run_candidate", run_candidate), ("compare", compare), ("accept", accept),
                     ("reject", reject), ("stop_at_limit", stop_at_limit)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "run_baseline")
    graph.add_edge("run_baseline", "diagnose")
    graph.add_conditional_edges("diagnose", lambda s: END if s.get("outcome") else "intervene")
    graph.add_conditional_edges("intervene", lambda s: END if s.get("outcome") else "run_candidate")
    graph.add_edge("run_candidate", "compare")
    graph.add_conditional_edges("compare", lambda s: "accept" if s["decision"]["accepted"] else "reject")
    graph.add_conditional_edges("accept", next_or_stop)
    graph.add_conditional_edges("reject", next_or_stop)
    graph.add_edge("stop_at_limit", END)
    return graph.compile()


def improve(*, adapter_cls, target: str, dataset: Path, cases, gates_config: dict, llm_call, approve,
            max_iterations: int = 3, dry_run: bool = False, baseline_run_id: str | None = None,
            primary: str = "task.success", concurrency: int = 2) -> ImproveState:
    for attr in ("prompt_store", "prompt_targets"):
        if not hasattr(adapter_cls, attr):
            raise ValueError(f"{adapter_cls.__name__} has no `{attr}`: sagwa improve needs a prompt-store adapter")
    graph = build_graph(adapter_cls=adapter_cls, target=target, dataset=dataset, cases=cases,
                        gates_config=gates_config, llm_call=llm_call, approve=approve,
                        max_iterations=max_iterations, dry_run=dry_run, primary=primary, concurrency=concurrency)
    initial: ImproveState = {"tried": [], "iteration": 0, "log": []}
    if baseline_run_id:
        # Reuse a stored run as the baseline (saves a full evaluation); its
        # prompt versions come from what that run recorded.
        from sagwa.storage import Run, get_session

        with get_session() as session:
            recorded = session.get(Run, baseline_run_id).prompt_version or ""
        initial["baseline_run_id"] = baseline_run_id
        initial["versions"] = dict(pair.split("=", 1) for pair in recorded.split(",") if pair) or \
            {n: adapter_cls.prompt_store.current(n) for n in adapter_cls.prompt_targets.values()}
    return graph.invoke(initial, {"recursion_limit": 10 * max_iterations + 10})
