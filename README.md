# Sagwa

**CI/CD infrastructure for LLM quality.** Run a golden dataset against any prompt/model/pipeline version, get back structured per-case metrics, and gate a merge automatically when quality regresses — the same way a failing unit test would.

Python 3.11+ · CLI built with Typer · Storage via SQLAlchemy + Alembic · Status: early alpha · License: Apache 2.0

## Overview

Teams shipping LLM-powered features today mostly deploy on vibes: someone tweaks a prompt, skims five outputs, and merges. There's no equivalent of a unit-test suite for "did this change make the AI worse at its job" — quality regresses silently, nobody notices until a customer complains, and there's no historical record of *when* or *why* it broke.

Sagwa closes that gap. It is not another AI app — it's testing and observability infrastructure that sits underneath AI apps, the same way pytest + CI sits underneath a normal codebase:

1. Define a **golden dataset** — versioned JSONL, reviewed like code — for an LLM task (RAG QA, summarization, classification).
2. Run it against **any target pipeline** via a stable adapter contract and get structured metrics back (accuracy, faithfulness, cost, latency).
3. **Diff** two runs and see exactly which cases regressed, with a statistically grounded significance call.
4. **Gate CI** so a PR that drops quality below a threshold fails the build, just like a broken test.
5. **Cluster failures** by semantic similarity to spot systemic issues instead of reading transcripts one by one.
6. Track **cost and latency** per pipeline version over time on a dashboard.

Full product spec: [docs/PRD.md](./docs/PRD.md). Technical plan and tech-stack rationale: [docs/PLAN.md](./docs/PLAN.md). As-built architecture: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

---

## Core Features & Capabilities

| Feature | How it works | Status |
|---|---|---|
| **Golden-set schema & loader** | Pydantic `GoldenCase` model (`id`, `input`, `expected_output`/`expected_labels`, `task_type`, `tags`); JSONL files are validated on load, with malformed rows rejected and reported by line number | Built and tested |
| **Pluggable target adapters** | Any pipeline implements `TargetAdapter.run(case_input) -> AdapterResult` (`answer`, `context`, `latency_ms`, `tokens`, `cost_usd`). Resolved at runtime via a built-in name (`stub`) or a `module.path:ClassName` string — no registry, no changes to Sagwa's source required to add a new target | Built and tested |
| **Async/concurrent eval runner** | Bounded-concurrency execution of a golden set against a target adapter (`--concurrency`); a failing case is captured per-row, not fatal to the run | Built and tested |
| **`sagwa run` CLI** | Wires loader, adapter, runner, metrics, and storage end-to-end; pins the target pipeline's own git SHA and model label when the adapter exposes them | Built and tested |
| **Run-history storage** | Append-only `runs`/`results` tables (SQLAlchemy + Alembic) pinning `sagwa_git_sha`, `target_pipeline_git_sha`, `dataset_sha256`, model, and status per run | Built and tested |
| **Reference-based & safety metrics** | Exact/fuzzy match, ROUGE-L, embedding similarity (where ground truth exists); PII-regex and toxicity-keyword flags per case | Built and tested |
| **LLM-as-judge harness** | Absolute and pairwise scoring modes, live-verified against Groq | Built and tested |
| **Judge calibration engine** | Cohen's kappa, confusion matrix, versioned calibration artifacts, refusal-to-gate below a kappa threshold, and a baseline-comparison mode (for scoring a prior judge, e.g. ringo's ad-hoc one, against the same human labels) | Built and tested against a synthetic fixture; real ~150-200-case human study not yet run |
| **ringo reference adapter** | Example adapter (`examples/adapters/ringo_adapter.py`) calling ringo's `pipeline.py` in-process (not HTTP — ringo's endpoint strips `context`, which faithfulness metrics need) | Implemented, unvalidated against a live ringo instance (its `documents/` corpus is currently empty) |
| **RAGAS metrics** (faithfulness, context precision/recall) | Wraps RAGAS for reference-free RAG scoring | Implemented but non-functional — `ragas==0.4.3` fails to import against installed `langchain-community==0.4.2`; degrades to `None` per metric rather than crashing the run |
| **`sagwa diff`** | Per-metric/per-tag regression detection with stat-sig testing | Not implemented — CLI stub that prints a message and exits 1 |
| **Failure clustering** | Embed failing cases, HDBSCAN cluster, auto-label | Not started — `sagwa/clustering/` is empty |
| **`sagwa gate` + CI Action** | Threshold-based CI gating from `config/gates.yaml` | Not implemented — CLI stub; `.github/workflows/eval-gate.yml` is a documented placeholder |
| **Dashboard** | Streamlit trend/cost/cluster browser | Not started — `sagwa/dashboard/` is empty |

**Summary:** the ingestion, run, metrics, and storage pipeline is production-tested MVP functionality. Diff, clustering, gating, and dashboard are explicit, documented roadmap items, not silently missing.

---

## Tech Stack & Architecture

| Layer | Choice |
|---|---|
| CLI | [Typer](https://typer.tiangolo.com/) — `sagwa run \| diff \| gate`, installed as a console script |
| Schema validation | Pydantic v2 |
| Storage | SQLAlchemy 2.0 + Alembic, **SQLite by default** (`sqlite:///./sagwa.db`) — see [ADR-0004](./docs/adr/0004-sqlite-default-defer-postgres.md); Postgres is a config change away, not a rewrite |
| Concurrency | Bounded thread pool (`sagwa/runner/`), no external task queue |
| LLM judge | Groq via `langchain-groq` (optional dependency group `judge`/`metrics`) |
| RAG metrics | RAGAS (optional, currently blocked on a dependency-version mismatch — see table above) |
| Clustering (planned) | `sentence-transformers` embeddings + HDBSCAN |
| Dashboard (planned) | Streamlit |
| Tracing | None yet — deferred, see [ADR-0002](./docs/adr/0002-defer-langfuse.md); `results.trace_id` is a reserved column |
| CI | GitHub Actions (`.github/workflows/eval-gate.yml`, currently a placeholder) |

### Data flow

```
golden_sets/*.jsonl  →  load_golden_set()  →  [GoldenCase, ...]
                                                    │
                                                    ▼
                              _load_adapter_class(target)().run(case.input)
                              ("stub" built in, or ANY "module.path:ClassName"
                               resolved dynamically at runtime)
                                                    │
                                                    ▼
                                             AdapterResult
                                {answer, context, latency_ms, tokens, cost_usd}
                                                    │
                                                    ▼
                                    Run / Result rows (sagwa/storage/models.py)
                                             persisted via get_session()
```

`sagwa/cli.py` has **no hardcoded knowledge of any specific target pipeline** — the only built-in adapter is `stub`. Everything else is resolved by importing a `module.path:ClassName` string at runtime, with the caller's `cwd` added to `sys.path` first (the way `python -m` would resolve it). This is what lets a completely separate project (e.g. [ringo](../ringo)) become a target pipeline with zero changes to either codebase.

### Directory guide

```
sagwa/
├── cli.py            # `sagwa run|diff|gate` entry points + adapter loading
├── adapters/          # TargetAdapter protocol + AdapterResult; stub only
├── datasets/           # GoldenCase pydantic schema + JSONL loader/validator
├── storage/            # SQLAlchemy models (Run, Result) + session factory
├── runner/             # bounded-concurrency case execution
├── metrics/            # reference.py, safety.py (live); ragas_metrics.py (blocked)
├── judge/              # harness.py (absolute/pairwise scoring); calibration.py
├── diff/               # empty — regression/stat-sig diff engine (roadmap)
├── clustering/          # empty — HDBSCAN failure clustering (roadmap)
└── dashboard/            # empty — Streamlit dashboard (roadmap)
examples/adapters/       # reference adapter implementations, outside sagwa/
├── README.md              # how to plug in any ML project as a target pipeline
└── ringo_adapter.py         # worked example: the ringo integration
golden_sets/            # versioned golden-set JSONL files
migrations/              # Alembic migrations for the run-history schema
calibration/             # judge calibration study artifacts
config/gates.yaml        # CI gate thresholds (not yet read by any code)
docs/                    # PRD.md, PLAN.md, ARCHITECTURE.md, adr/
```

---

## Prerequisites & Environment Setup

- **Python** ≥ 3.11
- **pip** (or any PEP 517-compatible installer)
- **git** (run records pin the current git SHA — required for `sagwa run` to resolve a non-`"unknown"` `sagwa_git_sha`)
- No database service to run — SQLite ships with the Python standard library

### Environment variables (`.env.example`)

| Variable | Required for | Description |
|---|---|---|
| `GROQ_API_KEY` | LLM-as-judge (`sagwa/judge/`) | The only external API this project calls directly. Uses the same provider as ringo's own ad-hoc judge, so the calibration head-to-head is apples-to-apples. |
| `DATABASE_URL` | Storage | Defaults to `sqlite:///./sagwa.db`. Swap for a Postgres DSN to switch stores — the schema is DB-agnostic (see [ADR-0004](./docs/adr/0004-sqlite-default-defer-postgres.md)). |

Target-pipeline-specific variables (e.g. `RINGO_REPO_PATH` for the example ringo adapter) are **not** Sagwa core config — each adapter owns and documents its own env vars. See [examples/adapters/README.md](./examples/adapters/README.md).

---

## Quickstart / Installation

```bash
git clone <this-repo-url> sagwa && cd sagwa

# Install with the optional dependency groups you need
pip install -e ".[metrics,judge,clustering,dashboard,dev]"

# Configure environment
cp .env.example .env
# then fill in GROQ_API_KEY if you'll use the judge harness

# Run database migrations — REQUIRED before tests or `sagwa run` on a fresh
# clone. A sagwa.db file existing does NOT mean it's migrated.
alembic upgrade head

# Run the test suite
pytest

# Run the bundled example golden set against the stub adapter
sagwa run --target stub --dataset golden_sets/example.jsonl
```

> **Gotcha:** `sagwa/storage/db.py` happily creates a `sagwa.db` file on first connection, but Alembic still has to run to create the `runs`/`results` tables. On a fresh clone (or after deleting `sagwa.db`), skipping `alembic upgrade head` fails tests with `no such table: runs`.

### Optional dependency groups

| Group | Adds | Used for |
|---|---|---|
| `postgres` | `psycopg[binary]` | Swapping `DATABASE_URL` to Postgres |
| `metrics` | `ragas`, `langchain-groq`, `sentence-transformers` | RAGAS RAG metrics + judge model calls + embedding similarity |
| `clustering` | `sentence-transformers`, `hdbscan`, `scipy` | Failure clustering (not yet implemented) |
| `dashboard` | `streamlit` | Dashboard (not yet implemented) |
| `dev` | `pytest` | Test suite |

---

## Usage Guide

### `sagwa run` — execute a golden set against a target pipeline

```bash
sagwa run --target <name-or-module:Class> --dataset <path-to-jsonl> [--concurrency N]
```

- `--target`: either the built-in `stub` adapter, or `module.path:ClassName` pointing at any class implementing `TargetAdapter` — resolved the same way `python -m` would from your current working directory.
- `--dataset`: path to a golden-set JSONL file (schema: `id`, `input`, `expected_output` or `expected_labels`, `task_type` ∈ `{rag_qa, summarization, classification}`, `tags`).
- `--concurrency`: max concurrent adapter calls (default `5`).

Each run is persisted (append-only) with its Sagwa git SHA, the target pipeline's git SHA (when the adapter exposes `repo_path`), a dataset content hash, and per-case results including latency, tokens, cost, and computed metrics.

**Example — bundled stub adapter:**

```bash
sagwa run --target stub --dataset golden_sets/example.jsonl
```

**Example — a custom adapter (e.g. ringo):**

```bash
sagwa run --target examples.adapters.ringo_adapter:RingoAdapter --dataset golden_sets/example.jsonl
```

### Writing your own adapter

Implement the `TargetAdapter` protocol anywhere on your filesystem — it never needs to live inside this repo:

```python
class TargetAdapter(Protocol):
    name: str
    def run(self, case_input: str) -> AdapterResult: ...
```

`AdapterResult` is `{answer, context, latency_ms, tokens, cost_usd}`. Your adapter owns its own configuration (API keys, repo paths) via its own `__init__`. Two optional duck-typed attributes let Sagwa pick up extra metadata without knowing which adapter it is:

- `self.repo_path` — pins the target's own git SHA into the run record.
- `self.model_label` — a human-readable model identifier, for targets that route across models/tiers instead of using one static model id.

See [examples/adapters/README.md](./examples/adapters/README.md) for the full walkthrough.

### `sagwa diff` and `sagwa gate` (planned)

```bash
sagwa diff --baseline <run_id> --candidate <run_id>   # not implemented — exits 1
sagwa gate --run-id <run_id> --config config/gates.yaml  # not implemented — exits 1
```

Both commands exist in the CLI today as explicit placeholders (see [docs/ARCHITECTURE.md §1](./docs/ARCHITECTURE.md)) so the intended interface is visible ahead of the implementation. Gate thresholds are already defined in version-controlled config (`config/gates.yaml`), e.g.:

```yaml
metrics:
  faithfulness:
    op: gte
    value: 0.85
```

`.github/workflows/eval-gate.yml` documents the intended CI shape but currently just echoes a TODO and exits `0`.

---

## Roadmap & Contributing

### Current status (2026-08-18)

Weeks 1–6 of the [12-week build plan](./docs/PLAN.md#9-timeline-1-3-months-solo) are done: golden-set schema/loader, run-history storage, adapter contract, async runner, reference/safety metrics, and the judge + calibration engine are all built and tested. The RAGAS integration and the ringo adapter are implemented but blocked/unvalidated (see the feature table above). The real ~150–200-case human calibration study — the project's credibility anchor — has not been run yet.

### Upcoming (from [docs/PLAN.md](./docs/PLAN.md))

| Weeks | Focus |
|---|---|
| 7–8 | Diff engine + statistical-significance testing; HDBSCAN failure clustering + auto-labeling |
| 9–10 | CI gate (GitHub Action + CLI), Langfuse tracing integration, cost/latency tracking |
| 11 | Streamlit dashboard (trend lines, cluster browser, run diff view) |
| 12 | Write-up, demo video, README polish |

See [docs/FUTURE_INITIATIVES.md](./docs/FUTURE_INITIATIVES.md) for scope explicitly deferred beyond v1 (online/production eval, multi-judge ensembling, hosted multi-tenant version, and others).

### Contributing

This is currently a solo portfolio/infrastructure project. If you're extending it:

1. Read the module's one-line docstring and the PRD FR-numbers it points to before touching an unimplemented module (`runner/`, `metrics/`, `judge/`, `diff/`, `clustering/`, `dashboard/`) — each was scaffolded against a specific spec section, not left empty by accident.
2. Check `docs/adr/` before "fixing" a load-bearing decision (SQLite default, in-process ringo integration, deferred tracing) — they're deliberate.
3. Run `alembic upgrade head` and `pytest` before and after any change.
4. Open an issue or PR describing the change against the relevant PRD/ARCHITECTURE section.

---

## License & Acknowledgments

Licensed under the [Apache License 2.0](./LICENSE).

Sagwa's primary real-world validation target is [ringo](../ringo), an independently maintained hybrid BM25+semantic RAG chat application, integrated purely through the external `TargetAdapter` contract with no shared code or coupling to its internals.
