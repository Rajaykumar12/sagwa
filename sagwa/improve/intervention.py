"""Propose a prompt revision for a diagnosed root cause (docs/AGENTIC_PRD.md
Phase 6). **Prompt text only** (locked decision #7): the output is a new
prompt string, validated and then written as a new version in the prompt
store. No code is ever generated or run.

The validation is the safety net, not the LLM's good behavior: a revision
that drops a `{placeholder}` the target fills in would crash or silently
blank part of the prompt, and a JSON prompt (tool descriptions) must stay
JSON over the same tools.
"""
from __future__ import annotations

import json
import re
import string

from sagwa.rootcause import RootCause

_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$")


def _placeholders(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def validate_revision(current: str, revised: str) -> list[str]:
    """Problems that disqualify `revised` as a replacement for `current`."""
    problems = []
    if not revised.strip():
        return ["empty revision"]
    if revised.strip() == current.strip():
        problems.append("identical to the current version")
    try:
        current_json = json.loads(current)
    except ValueError:
        current_json = None
    if isinstance(current_json, dict):
        try:
            revised_json = json.loads(revised)
        except ValueError:
            return problems + ["the current prompt is JSON and the revision isn't"]
        if not isinstance(revised_json, dict) or set(revised_json) != set(current_json):
            problems.append(f"must keep exactly these keys: {sorted(current_json)}")
        elif not all(isinstance(v, str) and v.strip() for v in revised_json.values()):
            problems.append("every value must be a non-empty string")
    else:
        try:
            missing, added = _placeholders(current) - _placeholders(revised), _placeholders(revised) - _placeholders(current)
        except ValueError as e:
            return problems + [f"not a valid format string: {e}"]
        if missing:
            problems.append(f"drops placeholder(s) {sorted(missing)}")
        if added:
            problems.append(f"adds placeholder(s) {sorted(added)} the target doesn't fill in")
    return problems


def revision_prompt(prompt_name: str, current: str, cause: RootCause, examples: list[str]) -> str:
    is_json = current.lstrip().startswith("{")
    rules = ("Return only a JSON object with exactly the same keys, each value a rewritten description."
             if is_json else
             "Keep every {placeholder} exactly as written (the code fills them in) and add no new ones. "
             "Return only the full revised prompt text.")
    return (
        f"You are improving the `{prompt_name}` prompt of an AI agent. An evaluation diagnosed this "
        f"failure:\n\nCause: {cause.category}\nWhat it means: {cause.hypothesis}\nEvidence:\n"
        + "".join(f"- {e}\n" for e in cause.evidence)
        + ("\nFailing cases:\n" + "".join(f"- {x}\n" for x in examples[:3]) if examples else "")
        + f"\nCurrent prompt:\n<<<\n{current}\n>>>\n\nRevise it so the agent avoids this failure. "
          f"Change only what addresses the cause. {rules}"
    )


def propose_revision(llm_call, prompt_name: str, current: str, cause: RootCause,
                     examples: list[str] | None = None) -> tuple[str | None, list[str]]:
    """`(revised_text, [])` for a valid revision, or `(None, problems)`."""
    revised = _FENCE.sub("", llm_call(revision_prompt(prompt_name, current, cause, examples or [])).strip())
    revised = revised.removeprefix("<<<").removesuffix(">>>").strip("\n")
    problems = validate_revision(current, revised)
    return (None, problems) if problems else (revised, [])
