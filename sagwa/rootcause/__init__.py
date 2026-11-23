"""Root-cause engine for agentic failures (docs/AGENTIC_PRD.md Phase 5).

Turns "7 cases failed" into "7 cases: HANDOFF — the researcher->writer
handoff passes on 18% of what the researcher found". Each failing case is
diagnosed from its trajectory into a **closed taxonomy**, then cases are
grouped by cause. A closed set of causes, chosen by rules rather than free-form
model prose, is what makes the result testable and gateable.

Two design choices, both forced by real data (2026-11-23):

- **Grouped by cause, not by HDBSCAN cluster.** HDBSCAN returns nothing below
  `min_cluster_size` failures, and a regression on a small PR golden set can
  be two cases. The taxonomy is itself the grouping; `sagwa cluster` still
  offers the unsupervised view.
- **An optional baseline run supplies the contrast.** Some causes only show
  against a known-good trajectory: a planner that stops planning a
  verification step drops it from *every* case, so the run's own passing
  cases look the same. With `baseline_run_id`, each case is compared with the
  same case in the baseline.

Success and trajectory metrics are recomputed from stored trajectories, not
read from `metrics_json`, so runs scored before the agent metrics existed
are diagnosable too.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sagwa.adapters.base import AgentRun
from sagwa.datasets.schema import GoldenCase
from sagwa.metrics.agent_success import check_success
from sagwa.metrics.agent_trajectory import compute_trajectory_metrics

# Precedence order: when several fire for one case, the first one is the cause.
# A loop explains running out of steps; answering what should have been
# declined explains whatever tools it then called; lost context explains a
# wrong answer.
CATEGORIES = ("INFINITE_RETRY", "BUDGET_EXHAUSTED", "SCOPE", "HANDOFF", "TOOL_SELECTION", "PLANNING", "GROUNDING",
              "UNKNOWN")

_LOOP_THRESHOLD = 0.3       # agent.loop_rate: 3rd+ identical calls as a share of all calls
_ACCURACY_FLOOR = 0.5       # agent.tool_call_accuracy below this is wrong-tool territory
_ACCURACY_DROP = 0.25       # ...or this much below the same case in the baseline


@dataclass
class CaseDiagnosis:
    case_id: str
    category: str
    signals: list[str]          # every category whose rule fired, in precedence order
    evidence: list[str]
    confidence: float
    # With a baseline: True when the same case also failed there. Such a case
    # is diagnosed on its own trajectory only, since a difference from a
    # baseline that failed too can't be what broke it. None without a baseline.
    pre_existing: bool | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RootCause:
    category: str
    case_ids: list[str]
    hypothesis: str
    evidence: list[str]
    confidence: float
    cases: list[CaseDiagnosis] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"category": self.category, "case_ids": self.case_ids, "hypothesis": self.hypothesis,
                "evidence": self.evidence, "confidence": self.confidence,
                "cases": [c.to_dict() for c in self.cases]}


def _agent_names(run: AgentRun) -> set[str]:
    return {s.name for s in run.trajectory if s.step_type == "agent"}


def diagnose_case(case: GoldenCase, run: AgentRun, baseline: AgentRun | None = None) -> CaseDiagnosis | None:
    """`None` when the case succeeded (or has no success criteria)."""
    if case.success_criteria is None:
        return None
    failed = check_success(case.success_criteria, run)
    if not failed:
        return None
    pre_existing = None if baseline is None else bool(check_success(case.success_criteria, baseline))
    if pre_existing:
        baseline = None  # see CaseDiagnosis.pre_existing
    m = compute_trajectory_metrics(run, case.expected_tools)
    b = compute_trajectory_metrics(baseline, case.expected_tools) if baseline else None
    tools = [s for s in run.trajectory if s.step_type == "tool"]
    fired: dict[str, list[str]] = {}

    if (m["loop_rate"] or 0) >= _LOOP_THRESHOLD:
        top, n = Counter((s.name, s.input) for s in tools).most_common(1)[0]
        errors = sum(1 for s in tools if (s.name, s.input) == top and s.error)
        fired["INFINITE_RETRY"] = [f"`{top[0]}` called {n}× with identical input, {errors} of them failed"]

    if "max_steps" in failed:
        limit = case.success_criteria.max_steps
        fired["BUDGET_EXHAUSTED"] = [f"{run.step_count} steps against a limit of {limit}"]

    if case.success_criteria.expect_refusal:
        fired["SCOPE"] = ["answered a request it should have declined"
                          + (f" (calling {', '.join(sorted({s.name for s in tools}))})" if tools else "")]

    lossy = [s for s in run.trajectory if s.step_type == "handoff" and s.input is not None and s.output != s.input]
    if lossy and not (b and (b["handoff_loss_rate"] or 0) > 0):
        h = lossy[0]
        kept = len(h.output or "") / max(len(h.input), 1)
        fired["HANDOFF"] = [f"`{h.name}` passed on {len(h.output or '')} of {len(h.input)} characters ({kept:.0%})"]

    accuracy = m["tool_call_accuracy"]
    dropped = accuracy is not None and b and b["tool_call_accuracy"] is not None \
        and b["tool_call_accuracy"] - accuracy >= _ACCURACY_DROP
    if {"required_tools", "forbidden_tools"} & set(failed) or (accuracy is not None and accuracy < _ACCURACY_FLOOR) \
            or dropped:
        used = sorted({s.name for s in tools}) or ["none"]
        fired["TOOL_SELECTION"] = [f"used {', '.join(used)}; expected {', '.join(case.expected_tools or []) or 'none'}"
                                   + (f" (accuracy {accuracy:.2f})" if accuracy is not None else "")]

    if baseline is not None:
        missing = sorted(_agent_names(baseline) - _agent_names(run))
        if missing:
            fired["PLANNING"] = [f"step(s) {', '.join(missing)} ran in the baseline and were skipped here"]

    answer_only = set(failed) <= {"answer_contains", "answer_matches"}
    if not fired and answer_only and not m["step_error"]:
        fired["GROUNDING"] = [f"tools ran without error, yet the answer failed {', '.join(failed)}"]

    signals = [c for c in CATEGORIES if c in fired] or ["UNKNOWN"]
    category = signals[0]
    # 1.0 when one rule explains the case, 0.7 when several competed, 0.5 for
    # GROUNDING (reached by elimination), 0 for UNKNOWN.
    confidence = {"GROUNDING": 0.5, "UNKNOWN": 0.0}.get(category, 1.0 if len(signals) == 1 else 0.7)
    evidence = fired.get(category, [f"failed {', '.join(failed)}; no trajectory signal"])
    return CaseDiagnosis(case.id, category, signals, evidence, confidence, pre_existing)


_HYPOTHESIS = {
    "INFINITE_RETRY": "The agent keeps retrying a failing tool instead of giving up: nothing is capping retries.",
    "BUDGET_EXHAUSTED": "Cases run out of steps before finishing.",
    "SCOPE": "The agent treats out-of-scope requests as in scope and answers them instead of declining.",
    "HANDOFF": "Context is lost between agents: a handoff passes on only part of what the sender produced.",
    "TOOL_SELECTION": "The agent calls the wrong tools. Check whether the tools' names and descriptions tell them apart.",
    "PLANNING": "The plan no longer includes a step the baseline ran, so that step is skipped.",
    "GROUNDING": "Tools ran cleanly but the answer is still wrong: the agent misreads or miscomputes from its evidence.",
    "UNKNOWN": "No trajectory signal explains these failures.",
}


def group_diagnoses(diagnoses: list[CaseDiagnosis]) -> list[RootCause]:
    """One `RootCause` per category, largest first."""
    by_category: dict[str, list[CaseDiagnosis]] = {}
    for d in diagnoses:
        by_category.setdefault(d.category, []).append(d)
    causes = []
    for category, cases in by_category.items():
        counted = Counter(line for d in cases for line in d.evidence)
        evidence = [f"{line}" + (f" ({n} cases)" if n > 1 else "") for line, n in counted.most_common(5)]
        causes.append(RootCause(
            category=category, case_ids=sorted(d.case_id for d in cases), hypothesis=_HYPOTHESIS[category],
            evidence=evidence, confidence=round(sum(d.confidence for d in cases) / len(cases), 2),
            cases=sorted(cases, key=lambda d: d.case_id),
        ))
    return sorted(causes, key=lambda c: (-len(c.case_ids), CATEGORIES.index(c.category)))


def root_causes(session, run_id: str, baseline_run_id: str | None = None,
                golden_cases: dict[str, GoldenCase] | None = None) -> list[RootCause]:
    """Entry point: diagnose every failing agentic case of `run_id`, optionally
    against the same cases in `baseline_run_id`, and group by cause.
    `golden_cases` defaults to the run's own dataset."""
    from sagwa.datasets import load_golden_set
    from sagwa.storage import Result, Run

    if golden_cases is None:
        golden_cases = {c.id: c for c in load_golden_set(session.get(Run, run_id).dataset_path)}
    baseline = {}
    if baseline_run_id is not None:
        baseline = {r.case_id: r.to_agent_run()
                    for r in session.query(Result).filter(Result.run_id == baseline_run_id, Result.error.is_(None))}
    diagnoses = []
    for result in session.query(Result).filter(Result.run_id == run_id, Result.error.is_(None)):
        case = golden_cases.get(result.case_id)
        if case is None:
            continue
        d = diagnose_case(case, result.to_agent_run(), baseline.get(result.case_id))
        if d is not None:
            diagnoses.append(d)
    return group_diagnoses(diagnoses)


def format_report(causes: list[RootCause]) -> str:
    if not causes:
        return "No failing agentic cases."
    total = sum(len(c.case_ids) for c in causes)
    lines = [f"Root causes for {total} failing case(s):"]
    for c in causes:
        old = sum(1 for d in c.cases if d.pre_existing)
        note = f" ({old} also failed in the baseline)" if old else ""
        lines += ["", f"{c.category} — {len(c.case_ids)} case(s){note}, confidence {c.confidence:.2f}",
                  f"  {c.hypothesis}"] + [f"  - {e}" for e in c.evidence] + [f"  cases: {', '.join(c.case_ids)}"]
    return "\n".join(lines)
