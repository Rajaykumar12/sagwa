"""Golden-set schema (PRD FR-1, FR-3)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

TaskType = Literal["rag_qa", "summarization", "classification"]


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
