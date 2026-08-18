"""Example TargetAdapter implementation: ringo (../ringo).

This is a reference integration, not part of Sagwa's core — it lives here,
outside the installable `sagwa` package, to demonstrate how any ML project
plugs into Sagwa: implement `TargetAdapter` (`sagwa/adapters/base.py`), then
point Sagwa at it with `sagwa run --target
examples.adapters.ringo_adapter:RingoAdapter` (run from this repo's root).
See examples/adapters/README.md for the general pattern.

Deliberately in-process rather than over HTTP: ringo's `/chat/text` endpoint
pops `context` out of the response before returning it (see
`ringo/backend/main.py`, `text_chat()`), so faithfulness/context-precision
metrics would have nothing to score against over the wire. Calling
`PipelineOrchestrator.process_text()` directly keeps the full result dict,
including context. See docs/adr/0003-ringo-adapter-in-process.md.

Known limitation, not a bug in this adapter: ringo's `pipeline.py`/`rag.py`
never compute token counts or cost anywhere, so `AdapterResult.tokens` and
`.cost_usd` stay `None` for this adapter rather than being estimated or
faked. `latency_ms` is measured here, adapter-side, since it's the one
timing signal ringo doesn't already report itself.
"""
import os
import sys
import time
from pathlib import Path

from sagwa.adapters.base import AdapterResult


class RingoAdapter:
    name = "ringo"

    def __init__(self, repo_path: str | None = None):
        # This adapter owns its own configuration — Sagwa's core doesn't
        # know or care that RINGO_REPO_PATH exists; that's this file's
        # concern alone.
        self.repo_path = Path(repo_path or os.environ["RINGO_REPO_PATH"])
        sys.path.insert(0, str(self.repo_path / "backend"))
        # Constructing PipelineOrchestrator loads a Whisper model (see
        # ringo/backend/pipeline.py's AudioInputProcessor), so it's deferred
        # until the first `.run()` call rather than paid in `__init__` —
        # registering this adapter shouldn't be a heavyweight operation.
        self._orchestrator = None
        # ringo routes per-query across model tiers (see rag.py's
        # `model_tier`), so there's no single static model id to report —
        # cli.py reads this via duck-typing (see `_run_model_label`).
        self.model_label = "ringo (tiered, see this file's docstring)"

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from pipeline import PipelineOrchestrator  # ringo's backend/pipeline.py

            self._orchestrator = PipelineOrchestrator()
        return self._orchestrator

    def run(self, case_input: str) -> AdapterResult:
        orchestrator = self._get_orchestrator()

        start = time.perf_counter()
        result = orchestrator.process_text(case_input)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if not result.get("success", True):
            raise RuntimeError(f"ringo pipeline reported failure: {result}")

        return AdapterResult(
            answer=result["response"],
            context=result.get("context") or None,
            latency_ms=latency_ms,
            tokens=None,
            cost_usd=None,
        )
