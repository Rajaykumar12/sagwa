# Sagwa

**LLM Evaluation & Regression-Testing Platform** — CI/CD infrastructure for LLM quality. Run a golden dataset against a target LLM pipeline, diff two runs for statistically-grounded regressions, cluster failures automatically, and gate a CI pipeline on quality thresholds.

See [docs/PRD.md](./docs/PRD.md) for the product spec and [docs/PLAN.md](./docs/PLAN.md) for the technical architecture and week-by-week build plan. Primary target pipeline is [../ringo](../ringo), a real, separately-maintained RAG chat app; Sagwa evaluates it through a thin adapter (`sagwa/adapters/ringo.py`) with zero coupling to its internals.

## Status

Scaffolding + Week 1-2 foundations (docs/PLAN.md §9): golden-set schema/loader, SQLite-backed run history, the adapter contract, and a minimal CLI wired end-to-end against a stub target. Everything past that (RAGAS metrics, the judge/calibration harness, diff engine, clustering, CI gate, dashboard) is scaffolded as empty modules with roadmap docstrings, not yet implemented.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[metrics,judge,clustering,dashboard,dev]"   # core deps + every optional group
cp .env.example .env                                         # fill in GROQ_API_KEY
alembic upgrade head                                          # creates sagwa.db (SQLite) locally
```

No services to start — storage defaults to a local SQLite file (`sagwa.db`). See [docs/adr/0004-sqlite-default-defer-postgres.md](./docs/adr/0004-sqlite-default-defer-postgres.md) for the Postgres migration path once one's needed.

## Try it

```bash
sagwa run --target stub --dataset golden_sets/example.jsonl
```

## Tests

```bash
pytest
```

## Local-first, one exception

Everything runs locally — a local SQLite file for run history, local embeddings for later clustering, no cloud dependency. The one external API is the LLM judge, on Groq's free tier (the same provider `ringo` already uses for its own ad-hoc judge, `backend/eval.py`) — see `.env.example`.

## Project layout

```
sagwa/          # the package: cli, datasets, adapters, runner, metrics, judge, diff, clustering, storage, dashboard
golden_sets/    # versioned golden datasets (JSONL)
migrations/     # Alembic migrations for the run-history schema (SQLite by default)
calibration/    # judge calibration study artifacts
config/         # CI gate thresholds (config/gates.yaml)
docs/           # PRD.md, PLAN.md, writeup.md, adr/ (architecture decision records)
```

See docs/PLAN.md §10 for the full rationale behind this layout.
