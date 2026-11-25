# ADR 0010: agent trajectories are stored one row per step

**Status**: Accepted (2026-11-25; the table shipped 2026-11-19 in migration 0003)

## Context

Evaluating an agent means evaluating what it did, not just what it answered (PRD §5.10). Each case's trajectory (agent turns, tool calls, handoffs, each with input, output, latency, tokens, cost and any error) had to be stored with the case's result. Two shapes were on the table:

1. **A JSON column on `results`** holding the whole trajectory.
2. **A `trace_steps` table**, one row per step, keyed to `results.id` and ordered by `seq`.

A third option was already reserved: `results.trace_id`, a plain column kept for an external tracer (Langfuse), whose integration [ADR-0002](./0002-defer-langfuse.md) deferred.

## Decision

**Option 2: the `trace_steps` table.** And **`results.trace_id` stays reserved for external tracing**, rather than being reused to link to these rows.

- **Steps get queried, not just displayed.** The trace viewer loads one case's steps in order. Root-cause analysis and the trajectory metrics read step types, names, errors and tool arguments. Questions like "every failed `shipment_tracker` call across a run" are a `WHERE` clause on a table, but parse-every-blob work on a JSON column.
- **The schema stays small and stable.** The step columns are the ones every agent has (type, name, input, output, latency, tokens, cost, error). Everything framework-specific (tool arguments, the model, the prompt version) goes in `metadata_json`. A new kind of detail needs no migration.
- **Old runs need nothing.** The migration is additive with no backfill: a run from before agents existed simply has no steps, and a non-agent adapter writes none.
- **The trajectory is the single source of truth.** Counts such as `step_count` or tool calls are derived from the steps, never stored next to them, so they can't disagree. The plan's separate tool-call and agent-invocation lists were dropped for the same reason.
- **Why not reuse `trace_id`.** Steps join on `result_id`, so a second link is unnecessary. Keeping `trace_id` free means a future Langfuse (or OpenTelemetry) integration can record the external trace id next to Sagwa's own steps, which is how the two would be cross-referenced. Sagwa's steps are the evaluation record; an external tracer is the debugging record. Neither replaces the other.

## Consequences

- Metrics are pure functions of stored steps (`Result.to_agent_run()` rebuilds a case), so any run can be rescored or diagnosed later with no model call. This is what let the trajectory metrics and root-cause engine be verified against runs recorded before they existed.
- A trajectory costs one row per step: a 41-case agent run is about 330 rows (8 steps per case on the reference agent), which is nothing for SQLite or Postgres ([ADR-0004](./0004-sqlite-default-defer-postgres.md)).
- Steps are linear (`seq`). Parallel tool calls are recorded in the order they completed; branching or nested sub-agents would need a parent pointer, a column to add if a target ever needs one.
