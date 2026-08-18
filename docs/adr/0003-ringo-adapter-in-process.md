# ADR 0003: ringo adapter calls `pipeline.py` in-process, not the HTTP API

**Status**: Accepted (2026-08-18)

## Context

PRD FR-3b says the reference v1 adapter integrates ringo "by calling its FastAPI endpoint." But `ringo/backend/main.py`'s `/chat/text` handler pops `context` out of the result dict before returning JSON (`context = result.pop("context", "")`, in the non-streaming branch of `text_chat()`), so an HTTP-only adapter would never see the retrieved context — which faithfulness and context-precision metrics need (PRD FR-9).

PLAN.md §6 already anticipates this, describing the adapter as calling ringo's endpoint "(or importing `pipeline.py` directly)."

## Decision

`sagwa/adapters/ringo.py` imports and calls `ringo/backend/pipeline.py`'s `PipelineOrchestrator.process_text(...)` in-process (adds `ringo/backend` to `sys.path` via `RINGO_REPO_PATH`), rather than calling `/chat/text` over HTTP.

## Consequences

- Zero changes to ringo's own code (still satisfies FR-3a's "zero required changes to the target's own codebase") — this only reaches into what ringo already exposes as an importable module.
- Sagwa's process needs ringo's `backend/` dependencies importable (or at least `pipeline.py`'s own import chain) at adapter-run time, not just network access to a running ringo server.
- If ringo's `process_text` return shape changes, the adapter needs updating — this is the "expected integration maintenance" risk PRD §9 already calls out, not a new one.
- The adapter is left as a skeleton in this scaffolding pass (`RingoAdapter.run` raises `NotImplementedError`) — the exact field mapping needs validating against a live ringo instance, which is Week 3-4 work, not this pass.
