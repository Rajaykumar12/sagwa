# Sagwa

**LLM Evaluation & Regression-Testing Platform** — CI/CD infrastructure for LLM quality. Run a golden dataset against a target LLM pipeline, diff two runs for statistically-grounded regressions, cluster failures automatically, and gate a CI pipeline on quality thresholds.

Primary target pipeline is [../ringo](../ringo), a real, separately-maintained RAG chat app; Sagwa evaluates it through a thin adapter (`sagwa/adapters/ringo.py`) with zero coupling to its internals.

> This README is the only Markdown file tracked in git (see [Docs](#docs) below for why) — it's meant to be the single source of truth for anyone browsing the repo without a local checkout of the working notes.

## Status

**Week 1-2 of 12** (of the build plan). Done: golden-set schema/loader, SQLite-backed run history + Alembic migration, the target-adapter contract, and a minimal CLI wired end-to-end against a stub target — all tested (`pytest`, 4/4 passing). Not started: RAGAS/judge metrics, judge calibration, the diff engine, failure clustering, the CI gate, and the dashboard — these exist only as empty modules with roadmap docstrings (`sagwa/{runner,metrics,judge,diff,clustering,dashboard}/`) or explicit CLI/workflow placeholders (`sagwa diff`, `sagwa gate`, `.github/workflows/eval-gate.yml`) that print "not implemented" rather than fake a result.

The `RingoAdapter` (`sagwa/adapters/ringo.py`) is a skeleton — `.run()` raises `NotImplementedError` and it isn't registered in the CLI yet, deliberately, until it's validated against a live ringo instance.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[metrics,judge,clustering,dashboard,dev]"   # core deps + every optional group
cp .env.example .env                                         # fill in GROQ_API_KEY
alembic upgrade head                                          # creates the runs/results tables in sagwa.db
```

No services to start — storage defaults to a local SQLite file (`sagwa.db`, gitignored). Postgres remains the intended production-shaped store once failure clustering needs `pgvector`; switching is a `DATABASE_URL` change plus `pip install -e ".[postgres]"`, no code changes.

**Gotcha**: `alembic upgrade head` must run before `pytest` or `sagwa run` — a `sagwa.db` file existing doesn't mean it's migrated (SQLAlchemy will happily create an empty file on first connection). Skipping this fails with `sqlite3.OperationalError: no such table: runs`.

## Try it

```bash
sagwa run --target stub --dataset golden_sets/example.jsonl
```

`sagwa diff --baseline <id> --candidate <id>` and `sagwa gate --run-id <id>` are registered CLI commands but not implemented yet.

## Tests

```bash
pytest
```

## Environment variables

See `.env.example` for the full list with rationale. Summary:

| Variable | Default | Used by |
|---|---|---|
| `GROQ_API_KEY` | — | The LLM judge (not wired up yet) — same provider ringo's own ad-hoc judge (`backend/eval.py`) uses, for an apples-to-apples calibration comparison later |
| `DATABASE_URL` | `sqlite:///./sagwa.db` | `sagwa/storage/db.py` |
| `RINGO_REPO_PATH` | `../ringo` | `sagwa/adapters/ringo.py`, once implemented |

## Project layout

```
sagwa/
├── cli.py            # `sagwa run|diff|gate` entry points
├── adapters/          # TargetAdapter protocol + AdapterResult; stub adapter (done), ringo adapter (skeleton)
├── datasets/           # GoldenCase pydantic schema + JSONL loader/validator
├── storage/            # SQLAlchemy models (Run, Result) + session factory
├── runner/             # empty — async batch execution, not started
├── metrics/            # empty — reference-based/RAGAS/safety metrics, not started
├── judge/              # empty — LLM-as-judge + calibration workflow, not started
├── diff/               # empty — regression/stat-sig diff engine, not started
├── clustering/          # empty — HDBSCAN failure clustering, not started
└── dashboard/            # empty — Streamlit dashboard, not started
golden_sets/            # versioned golden-set JSONL files (currently: 3-case example.jsonl)
migrations/              # Alembic migrations for the run-history schema
calibration/             # judge calibration study artifacts (currently: stub report)
config/gates.yaml        # CI gate thresholds — not yet read by any code
tests/                   # pytest suite (dataset loader + storage roundtrip)
docs/                    # working notes — see below
```

## Docs

Everything under `docs/` (plus root `CLAUDE.md` and `calibration/calibration_report.md`) is **gitignored on purpose** — local working notes, not part of the public repo history. They still exist on disk in any checkout that generated them; a fresh `git clone` won't have them.

- `docs/PRD.md` — product requirements: goals, personas, functional requirements, success metrics, v1/v2 scope split
- `docs/PLAN.md` — technical architecture, tech-stack rationale, week-by-week build plan
- `docs/ARCHITECTURE.md` — the system **as it actually exists today**: current-vs-planned status table, real tech stack, directory guide, key contracts, env vars, build/test commands
- `docs/FUTURE_INITIATIVES.md` — itemized v2/future scope, deliberately deferred past v1
- `docs/adr/` — architecture decision records for every point where the code deviates from PLAN.md (e.g. SQLite instead of Postgres for now, in-process ringo integration instead of HTTP, deferring Langfuse)
- `docs/writeup.md` — stub for the eventual portfolio write-up
- `calibration/calibration_report.md` — stub for the judge calibration study (the project's critical-path deliverable, not started yet)

## Tech stack (current, see docs/ARCHITECTURE.md for full rationale)

Typer (CLI) · Pydantic v2 (schema validation) · SQLAlchemy 2.0 + Alembic (storage, SQLite by default) · pytest (tests). RAGAS, a Groq-backed judge, `sentence-transformers` + HDBSCAN (clustering), and Streamlit (dashboard) are declared as optional dependency groups in `pyproject.toml` but not yet used by any code.
