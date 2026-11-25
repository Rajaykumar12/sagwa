# ADR 0009: Sagwa ships an MCP server

**Status**: Accepted (2026-11-25). Revisits the MCP deferral in [ADR-0006](./0006-ci-baseline-via-artifact.md).

## Context

ADR-0006 deferred MCP because it would be "another way to reach the same `run`/`diff`/`gate` functions, with no named client", while the missing CI baseline was what blocked adoption. Both conditions have changed:

- The baseline exists and is verified on real PRs (2026-11-19).
- There is a named client. Coding agents (Claude Code and other MCP hosts) now make the changes Sagwa evaluates, so the agent that changed a prompt is the one best placed to ask whether it regressed anything. The agentic plan (now PRD §5.10, FR-40) made MCP a real phase for this reason.

## Decision

**`sagwa mcp` serves eight tools over stdio**, each a thin wrapper over the function its CLI command calls: `run_evaluation`, `get_run`, `compare_runs`, `check_gate`, `inspect_failure`, `get_trace`, `cluster_failures`, `root_causes`.

- **No evaluation logic in the MCP layer.** Each tool returns its function's own `to_dict()`, so the CLI and MCP can't drift apart.
- **stdio only.** The server runs as a local subprocess of the client, next to the run database. An HTTP transport would need authentication and a deployment story that nobody has asked for yet.
- **Expected errors reach the agent.** An unknown run, a missing file or a bad config is raised as the SDK's `ToolError`, so the message gets through. Anything else stays a crash whose details stay in the server's log.
- **MCP SDK v2 (`mcp>=2.0`), as an optional extra.** The core install doesn't need it.

## Consequences

- An agent can run the full "did my change regress it?" loop (run → gate → diff → root cause → trace) with tool calls alone. Verified with an in-memory client in tests and with a real stdio client against `sagwa mcp`.
- `run_evaluation` runs the target, and the judge when a key is set, so it can be slow and spend tokens. That's reported in its description, so the agent knows.
- Building it surfaced an existing bug: an **empty** `GROQ_API_KEY` counted as configured, so every case tried (and retried) a call that couldn't succeed. Empty now means unset everywhere a key is checked.
