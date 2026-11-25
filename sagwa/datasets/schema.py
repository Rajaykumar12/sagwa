"""Golden-set schema (PRD FR-1, FR-3; agentic cases: PRD FR-31)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

TaskType = Literal["rag_qa", "summarization", "classification", "agentic"]


class SuccessCriteria(BaseModel):
    """What counts as an agentic case succeeding, checked deterministically
    against the final answer and the trajectory: never by the LLM judge, and
    never via the gate config (that would make success depend on a gate).

    Every listed condition must hold. Matching is case-insensitive."""

    answer_contains: list[str] = Field(default_factory=list)
    answer_matches: Optional[str] = None  # a regex, searched (not full-matched)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    max_steps: Optional[int] = None
    must_not_error: bool = True  # no step in the trajectory may record an error
    # The right answer is to decline (an out-of-scope request). Not a check of
    # its own, since `answer_matches` says what a refusal looks like; it tells
    # root-cause analysis that a failure here is a scope failure.
    expect_refusal: bool = False


class GoldenCase(BaseModel):
    """One row of a golden dataset.

    `id` must be stable across dataset revisions — it's the join key used
    later to diff two runs of the same case (PRD FR-17).
    """

    id: str
    input: str
    expected_output: Optional[str] = None
    expected_labels: Optional[list[str]] = None
    task_type: TaskType
    tags: list[str] = Field(default_factory=list)
    # Agentic cases only. `expected_tools` is the tool set a good trajectory
    # uses (for tool-selection accuracy), separate from the hard pass/fail
    # conditions in `success_criteria`.
    success_criteria: Optional[SuccessCriteria] = None
    expected_tools: Optional[list[str]] = None

    @model_validator(mode="after")
    def _agentic_cases_define_success(self) -> "GoldenCase":
        if self.task_type == "agentic" and self.success_criteria is None:
            raise ValueError("an 'agentic' case must define success_criteria")
        return self
