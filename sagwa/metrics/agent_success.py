"""`task.success`: did an agentic case succeed? (PRD FR-32)

Checked against the case's own `success_criteria`, deterministically: never by
the LLM judge, and never via the gate config, which would make "success"
depend on the thresholds it is gated against (PRD FR-32).
"""
from __future__ import annotations

import re
import unicodedata

from sagwa.adapters.base import AgentRun
from sagwa.datasets.schema import SuccessCriteria

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def normalize_answer(text: str) -> str:
    """Typography off, wording untouched: every Unicode space becomes a plain
    space and every dash a hyphen. Models write "5\u202fyears" (a narrow
    no-break space) and "2026\u201110\u201101"; without this, four correct
    answers in the first full agent run were graded wrong."""
    text = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in text or "")
    return text.translate(_DASHES)


def check_success(criteria: SuccessCriteria, run: AgentRun) -> list[str]:
    """The conditions `run` fails, by name. Empty means success."""
    raw = normalize_answer(run.answer)
    answer = raw.lower()
    tools_used = [s.name for s in run.trajectory if s.step_type == "tool"]
    failed = []
    if any(text.lower() not in answer for text in criteria.answer_contains):
        failed.append("answer_contains")
    if criteria.answer_matches and not re.search(criteria.answer_matches, raw, re.IGNORECASE):
        failed.append("answer_matches")
    if any(tool not in tools_used for tool in criteria.required_tools):
        failed.append("required_tools")
    if any(tool in tools_used for tool in criteria.forbidden_tools):
        failed.append("forbidden_tools")
    if criteria.max_steps is not None and run.step_count > criteria.max_steps:
        failed.append("max_steps")
    if criteria.must_not_error and any(s.error for s in run.trajectory):
        failed.append("must_not_error")
    return failed


def compute_success(criteria: SuccessCriteria, run: AgentRun) -> dict:
    """`{"success": bool, "failed": [condition, ...]}` — the failed list is
    stored too, so a failing case says which condition it missed."""
    failed = check_success(criteria, run)
    return {"success": not failed, "failed": failed}
