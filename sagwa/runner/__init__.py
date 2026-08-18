"""Async batch eval-runner: concurrent/rate-limited execution of a golden set
against a target adapter (PRD FR-5, FR-6).

Adapters are synchronous (`TargetAdapter.run() -> AdapterResult`), so
concurrency is provided via a bounded thread pool rather than asyncio — this
is the "simple httpx + semaphore"-equivalent option PLAN.md §5 calls out,
right-sized for adapters that are themselves blocking calls (HTTP, or an
in-process ringo pipeline call). `max_concurrency` is the configurable rate
limit (FR-5).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from sagwa.adapters.base import AdapterResult, TargetAdapter
from sagwa.datasets.schema import GoldenCase


@dataclass
class CaseOutcome:
    """Result of running one golden-set case: either `result` is populated,
    or `error` is — never both. A raised adapter exception is captured here
    rather than propagated, so one bad case doesn't discard results for
    every other case in the run."""

    case: GoldenCase
    result: AdapterResult | None
    error: str | None


def run_cases(
    adapter: TargetAdapter,
    cases: list[GoldenCase],
    max_concurrency: int = 5,
) -> list[CaseOutcome]:
    """Execute `cases` against `adapter`, `max_concurrency` at a time."""

    def _run_one(case: GoldenCase) -> CaseOutcome:
        try:
            return CaseOutcome(case=case, result=adapter.run(case.input), error=None)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any adapter
            # failure becomes a recorded per-case error, not a crashed run.
            return CaseOutcome(case=case, result=None, error=f"{type(e).__name__}: {e}")

    outcomes_by_case_id: dict[str, CaseOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(_run_one, case) for case in cases]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes_by_case_id[outcome.case.id] = outcome

    # as_completed() yields in completion order; restore dataset order so
    # results are stable and diffable regardless of concurrency timing.
    return [outcomes_by_case_id[case.id] for case in cases]
