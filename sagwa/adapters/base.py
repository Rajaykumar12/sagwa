"""Target-pipeline adapter contract (PRD FR-3a).

Any target pipeline is integrated by implementing `TargetAdapter` — zero
required changes to the target's own codebase.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class AdapterResult:
    answer: str
    context: str | None
    latency_ms: int
    tokens: int | None
    cost_usd: float | None


class TargetAdapter(Protocol):
    name: str

    def run(self, case_input: str) -> AdapterResult:
        ...
