"""Trivial in-memory adapter, used to validate the runner/storage plumbing
before any real target pipeline is wired up (PLAN.md §13)."""
import time

from sagwa.adapters.base import AdapterResult


class StubAdapter:
    name = "stub"

    def run(self, case_input: str) -> AdapterResult:
        start = time.time()
        answer = f"[stub echo] {case_input}"
        return AdapterResult(
            answer=answer,
            context=None,
            latency_ms=int((time.time() - start) * 1000),
            tokens=len(case_input.split()),
            cost_usd=0.0,
        )
