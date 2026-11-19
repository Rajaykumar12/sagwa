# ADR 0004: SQLite as the default local store, defer Postgres/Docker

**Status**: Accepted (2026-08-18)

## Context

The scaffold originally defaulted `DATABASE_URL` to Postgres via `docker-compose.yml` (`pgvector/pgvector:pg16`), per PLAN.md §5's storage choice. Pulling that image and installing the full dependency set both took a long time in this environment, and the priority right now is proving the CLI → schema → storage path actually runs end-to-end, not standing up production-shaped infra first.

`sagwa/storage/db.py` and `migrations/env.py` both already just read `DATABASE_URL` from the environment and hand it to SQLAlchemy/Alembic — nothing in `sagwa/storage/models.py` is Postgres-specific.

## Decision

Default `DATABASE_URL` to `sqlite:///./sagwa.db` (no service to run, no driver to install — `sqlite3` is in the Python standard library). `docker-compose.yml` is removed. `psycopg[binary]` moves from a core dependency to an optional `postgres` extra (`pip install -e ".[postgres]"`).

## Consequences

- Zero external services needed to run `sagwa run`, `alembic upgrade head`, or the test suite — the whole base prototype works offline with nothing but the venv.
- This is **not** a reversal of the architecture: Postgres (+ pgvector, once Week 7-8 clustering needs to store failure-case embeddings — PLAN.md's "one fewer moving part" rationale) is still the intended production-shaped storage. Switching back is: set `DATABASE_URL` to a Postgres URL, `pip install -e ".[postgres]"`, and re-run migrations — no code changes, since the schema and Alembic setup are already database-agnostic.
- SQLite has no `pgvector` equivalent, so a Postgres or a separate vector store is still required before failure-clustering work (Week 7-8) can store embeddings in the DB itself — that decision is deferred to when that work starts, not made here.
