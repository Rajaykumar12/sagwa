from sagwa.adapters.base import AdapterResult
from sagwa.datasets.schema import GoldenCase
from sagwa.runner import run_cases


class _FlakyAdapter:
    """Fails on cases whose input starts with 'fail', succeeds otherwise."""

    name = "flaky"

    def run(self, case_input: str) -> AdapterResult:
        if case_input.startswith("fail"):
            raise RuntimeError("boom")
        return AdapterResult(answer=f"ok: {case_input}", context=None, latency_ms=1, tokens=1, cost_usd=0.0)


def _case(case_id: str, text: str) -> GoldenCase:
    return GoldenCase(id=case_id, input=text, task_type="rag_qa")


def test_all_cases_succeed_preserves_order():
    cases = [_case("a", "one"), _case("b", "two"), _case("c", "three")]
    outcomes = run_cases(_FlakyAdapter(), cases, max_concurrency=3)

    assert [o.case.id for o in outcomes] == ["a", "b", "c"]
    assert all(o.error is None for o in outcomes)
    assert outcomes[0].result.answer == "ok: one"


def test_one_failing_case_does_not_abort_the_run():
    cases = [_case("a", "one"), _case("b", "fail this one"), _case("c", "three")]
    outcomes = run_cases(_FlakyAdapter(), cases, max_concurrency=3)

    by_id = {o.case.id: o for o in outcomes}
    assert by_id["a"].error is None
    assert by_id["c"].error is None
    assert by_id["b"].result is None
    assert "boom" in by_id["b"].error


def test_max_concurrency_is_respected():
    import threading
    import time

    peak = {"value": 0}
    current = {"value": 0}
    lock = threading.Lock()

    class _SlowAdapter:
        name = "slow"

        def run(self, case_input: str) -> AdapterResult:
            with lock:
                current["value"] += 1
                peak["value"] = max(peak["value"], current["value"])
            time.sleep(0.05)
            with lock:
                current["value"] -= 1
            return AdapterResult(answer="ok", context=None, latency_ms=1, tokens=None, cost_usd=None)

    cases = [_case(str(i), str(i)) for i in range(10)]
    run_cases(_SlowAdapter(), cases, max_concurrency=2)

    assert peak["value"] <= 2
