# Sagwa

CI/CD infrastructure for LLM quality. Sagwa runs a versioned golden dataset
against any LLM pipeline, computes structured per-case metrics, diffs two
runs for statistically-grounded regressions, clusters failures automatically,
and gates a CI pipeline on quality thresholds — the same role pytest + CI
plays for a normal codebase, applied to prompts, models, and RAG configs.

```
golden set (.jsonl)  →  sagwa run  →  metrics per case  →  sagwa diff  →  sagwa gate (CI)
```

## Why

A prompt tweak, a model swap, or a retrieval change today usually ships on
vibes: someone skims a handful of outputs and merges. Sagwa replaces that
with a testable, auditable process — every run is pinned to a git SHA and
dataset version, every regression claim comes with a significance test, and
every judge score is calibrated against human labels before it's trusted for
gating.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[metrics,judge,clustering,dashboard,dev]"
cp .env.example .env          # fill in GROQ_API_KEY
alembic upgrade head          # creates the runs/results tables — required before first run
```

Storage defaults to a local SQLite file (`sagwa.db`, gitignored) — no
services to start. `DATABASE_URL` switches to Postgres later with no code
changes (see `docs/adr/0004-sqlite-default-defer-postgres.md`).

## Usage

Run a golden set against the built-in stub target (validates the pipeline
end-to-end with no external dependencies):

```bash
sagwa run --target stub --dataset golden_sets/example.jsonl
```

### Evaluating your own pipeline

Sagwa integrates any ML project through one small contract —
`TargetAdapter` (`sagwa/adapters/base.py`) — with **zero required changes
to either codebase**: implement one class, point `--target` at it.

```python
# your_project/sagwa_adapter.py
from sagwa.adapters.base import AdapterResult

class MyAdapter:
    name = "my-pipeline"

    def run(self, case_input: str) -> AdapterResult:
        answer = my_pipeline.answer(case_input)
        return AdapterResult(answer=answer, context=None, latency_ms=..., tokens=..., cost_usd=...)
```

```bash
sagwa run --target your_project.sagwa_adapter:MyAdapter --dataset your_golden_set.jsonl
```

See [`examples/adapters/README.md`](examples/adapters/README.md) for the
full guide, and `examples/adapters/ringo_adapter.py` for a worked example
integrating [ringo](../ringo), a real RAG chat app.

### Diffing and gating

```bash
sagwa diff --baseline <run-id> --candidate <run-id>   # not yet implemented
sagwa gate --run-id <run-id>                          # not yet implemented
```

## Golden sets

A golden set is a versioned JSONL file, one case per line:

```json
{"id": "ex-001", "input": "What is the capital of France?", "expected_output": "Paris", "task_type": "rag_qa", "tags": ["geography"]}
```

Golden sets live in git alongside code — a change to expected behavior goes
through a PR and is diffable, the same review discipline as a test file.

## How it works

```
sagwa/
├── cli.py         # `sagwa run|diff|gate` + the adapter-loading mechanism
├── adapters/       # TargetAdapter protocol; the `stub` built-in
├── datasets/        # golden-set schema + JSONL loader/validator
├── runner/           # bounded-concurrency case execution
├── metrics/           # reference-based, safety, RAGAS metrics
├── judge/              # LLM-as-judge harness + calibration engine
├── diff/                 # regression detection + significance testing
├── clustering/            # failure clustering
├── dashboard/              # trend/cost/cluster views
└── storage/                 # SQLAlchemy models + migrations (Run, Result)
examples/adapters/    # reference TargetAdapter implementations (e.g. ringo)
golden_sets/           # versioned golden-set JSONL files
calibration/            # judge calibration study artifacts
config/gates.yaml        # CI gate thresholds
```

Each run is persisted as a `Run` row (git SHA, dataset version, model,
target) with one `Result` row per case (output, latency, cost,
`metrics_json`) — both tables are append-only, so run history is never
mutated, only appended.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | LLM judge + RAGAS metrics |
| `DATABASE_URL` | `sqlite:///./sagwa.db` | run-history storage |

These are the only two core env vars. Each target adapter owns its own
config (e.g. the ringo example's `RINGO_REPO_PATH`) — see
`examples/adapters/README.md`.

## Testing

```bash
pytest
```

## Status

Golden-set schema/loader, run-history storage, the pluggable adapter
mechanism, the async runner, reference-based + safety metrics, and the
LLM-judge + calibration engine are implemented and tested. The diff engine,
failure clustering, CI gate, and dashboard are not yet built. For the full
requirement-by-requirement breakdown, known blockers (a live target corpus,
a `ragas` dependency mismatch, real human calibration labels), and the
build order, see `docs/GAP_ANALYSIS.md` and `docs/ARCHITECTURE.md`.

## Docs

Everything under `docs/` (plus root `CLAUDE.md` and
`calibration/calibration_report.md`) is gitignored — local working notes,
present on disk in any checkout that generated them but not part of the
public repo history.

- `docs/PRD.md` — product requirements: goals, personas, functional requirements, scope
- `docs/PLAN.md` — technical architecture, tech-stack rationale, build plan
- `docs/ARCHITECTURE.md` — the system as it actually exists: status table, directory guide, key contracts
- `docs/GAP_ANALYSIS.md` — production-readiness gap matrix and build order
- `docs/FUTURE_INITIATIVES.md` — itemized v2/future scope
- `docs/adr/` — architecture decision records
- `examples/adapters/README.md` — how to plug in your own pipeline (tracked — part of the public example, not a working note)

## Tech stack

Typer (CLI) · Pydantic v2 (schema validation) · SQLAlchemy 2.0 + Alembic
(storage) · Groq (LLM judge) · `sentence-transformers` (embedding
similarity) · RAGAS (RAG metrics) · HDBSCAN + Streamlit (clustering,
dashboard — planned). See `docs/ARCHITECTURE.md` for the full rationale.
