"""A fixture adapter that is NOT registered anywhere in sagwa/'s source —
used by tests/test_cli_adapter_loading.py to prove a third-party adapter
loads via the dynamic `module:ClassName` mechanism alone."""
from sagwa.adapters.base import AdapterResult


class FakeAdapter:
    name = "fake"

    def run(self, case_input: str) -> AdapterResult:
        return AdapterResult(
            answer=f"fake: {case_input}", context=None, latency_ms=1, tokens=1, cost_usd=0.0
        )
