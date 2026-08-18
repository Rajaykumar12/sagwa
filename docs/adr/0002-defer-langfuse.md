# ADR 0002: Defer Langfuse tracing

**Status**: Accepted (2026-08-18)

## Context

PLAN.md §5 lists self-hosted Langfuse for span-level tracing and a trace UI. Langfuse is itself a multi-service stack (its own Postgres/ClickHouse/Redis), which is significant infra weight to add on top of Sagwa's own Postgres for a solo, local-first build — especially before any of the eval logic it would trace exists yet.

## Decision

Defer Langfuse. Sagwa's own `runs`/`results` tables (`sagwa/storage/models.py`) already capture per-case latency, tokens, cost, and a `trace_id` column, which is what PRD FR-6 actually requires. `results.trace_id` is left as a plain string column so a Langfuse (or any other tracer's) trace ID can be dropped in later without a schema change.

## Consequences

- No Langfuse service in `docker-compose.yml` for now.
- Revisit in the Week 9-10 slot from PLAN.md §9 if the dashboard/CI work later in the project wants Langfuse's UI specifically — at that point `trace_id` is already there to link against it.
