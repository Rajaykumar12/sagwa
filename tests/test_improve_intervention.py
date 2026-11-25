"""Prompt revisions are validated before anything is written: placeholders
kept, JSON prompts stay JSON over the same tools, and the text must change."""
import json

from sagwa.improve.intervention import propose_revision, revision_prompt, validate_revision
from sagwa.rootcause import RootCause

CAUSE = RootCause(category="PLANNING", case_ids=["c1"], hypothesis="A step is skipped.",
                  evidence=["step(s) reviewer ran in the baseline and were skipped here"], confidence=1.0)
PLANNER = "Write a plan for: {question}"
TOOLS = json.dumps({"search": "Look up info.", "lookup": "Look up info."})


def test_valid_text_revision():
    assert validate_revision(PLANNER, "Write a plan for: {question}. End with a Verify: step.") == []


def test_dropping_or_adding_a_placeholder_is_refused():
    assert validate_revision(PLANNER, "Write a plan.") == ["drops placeholder(s) ['question']"]
    assert validate_revision(PLANNER, "Plan {question} by {deadline}") == [
        "adds placeholder(s) ['deadline'] the target doesn't fill in"]


def test_an_unchanged_or_empty_revision_is_refused():
    assert validate_revision(PLANNER, PLANNER + "\n") == ["identical to the current version"]
    assert validate_revision(PLANNER, "  ") == ["empty revision"]


def test_json_prompts_must_stay_json_over_the_same_keys():
    good = json.dumps({"search": "Policy documents.", "lookup": "Orders database (SQL)."})
    assert validate_revision(TOOLS, good) == []
    assert validate_revision(TOOLS, "Use search for policies.") == ["the current prompt is JSON and the revision isn't"]
    assert validate_revision(TOOLS, json.dumps({"search": "x"})) == ["must keep exactly these keys: ['lookup', 'search']"]
    assert validate_revision(TOOLS, json.dumps({"search": "x", "lookup": ""})) == ["every value must be a non-empty string"]


def test_the_llm_sees_the_cause_evidence_and_current_prompt():
    text = revision_prompt("planner", PLANNER, CAUSE, ["Q: how many late? A: 11 (expected 13)"])
    assert "Cause: PLANNING" in text and "reviewer ran in the baseline" in text
    assert PLANNER in text and "Q: how many late?" in text and "{placeholder}" in text


def test_propose_revision_strips_fences_and_validates():
    fenced = lambda prompt: "```\nWrite a plan for: {question}. End with Verify:.\n```"
    revised, problems = propose_revision(fenced, "planner", PLANNER, CAUSE)
    assert problems == [] and revised == "Write a plan for: {question}. End with Verify:."
    broken = lambda prompt: "Write a plan."
    assert propose_revision(broken, "planner", PLANNER, CAUSE) == (None, ["drops placeholder(s) ['question']"])
