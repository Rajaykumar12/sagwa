# Sagwa

**CI/CD infrastructure for LLM quality.** Run a golden dataset against any prompt/model/pipeline version, get back structured per-case metrics, and gate a merge automatically when quality regresses, the same way a failing unit test would.

Python 3.11+. CLI built with Typer. Storage via SQLAlchemy and Alembic. Status: early alpha. License: Apache 2.0.

## Overview

Teams shipping LLM-powered features today mostly deploy on vibes: someone tweaks a prompt, skims five outputs, and merges. There's no equivalent of a unit-test suite for "did this change make the AI worse at its job" - quality regresses silently, nobody notices until a customer complains, and there's no historical record of when or why it broke.

Sagwa closes that gap. It is not another AI app, it's testing and observability infrastructure that sits underneath AI apps, the same way pytest and CI sit underneath a normal codebase:

1. Define a golden dataset - versioned JSONL, reviewed like code - for an LLM task (RAG QA, summarization, classification).
2. Run it against any target pipeline via a stable adapter contract and get structured metrics back (accuracy, faithfulness, cost, latency).
3. Diff two runs and see exactly which cases regressed, with a statistically grounded significance call.
4. Gate CI so a PR that drops quality below a threshold fails the build, just like a broken test.
5. Cluster failures by semantic similarity to spot systemic issues instead of reading transcripts one by one.
6. Track cost and latency per pipeline version over time on a dashboard.

## Core Features and Capabilities

| Feature | How it works | Status |
|---|---|---|
| Golden-set schema and loader | Pydantic `GoldenCase` model (`id`, `input`, `expected_output`/`expected_labels`, `task_type`, `tags`); JSONL files are validated on load, with malformed rows rejected and reported by line number | Built and tested |
| Pluggable target adapters | Any pipeline implements `TargetAdapter.run(case_input) -> AdapterResult` (`answer`, `context`, `latency_ms`, `tokens`, `cost_usd`). Resolved at runtime via a built-in name (`stub`) or a `module.path:ClassName` string, no registry, no changes to Sagwa's source required to add a new target | Built and tested |
| Async/concurrent eval runner | Bounded-concurrency execution of a golden set against a target adapter (`--concurrency`); a failing case is captured per-row, not fatal to the run | Built and tested |
| `sagwa run` CLI | Wires loader, adapter, runner, metrics, and storage end-to-end; persists each case as it finishes, so `--resume <run_id>` re-runs only what an interrupted run missed; pins the target pipeline's own git SHA and model label when the adapter exposes them | Built and tested |
| Run-history storage | Append-only `runs`/`results` tables (SQLAlchemy + Alembic) pinning `sagwa_git_sha`, `target_pipeline_git_sha`, `dataset_sha256`, model, and status per run | Built and tested |
| Reference-based, classification, and safety metrics | Exact/fuzzy match, ROUGE-L, embedding similarity (where a reference string exists); set-based precision/recall/F1 and exact-set-match (where `expected_labels` exists); PII-regex and toxicity-keyword flags on every case | Built and tested |
| LLM-as-judge harness | Absolute and pairwise scoring modes, wired into `sagwa run` itself (every case gets a judge score plus the judge's raw rationale text, not just reference metrics), live-verified against Groq | Built and tested |
| Judge calibration engine | Cohen's kappa, confusion matrix, versioned calibration artifacts, refusal-to-gate below a kappa threshold, and a baseline-comparison mode (for scoring a prior judge against the same human labels) | Built and measured: κ = 0.450 (95% CI [0.325, 0.569], accuracy 72.5%) against 200 human labels from HelpSteer2, below the 0.70 bar, so the gate refuses this judge |
| RAGAS metrics (faithfulness, context precision) | Wraps RAGAS for reference-free RAG scoring | Built and live-verified against Groq (a `ragas`/`langchain-community` version pin resolved the prior import failure) |
| `sagwa diff` | Per-metric/per-tag regression detection: paired bootstrap CI for continuous metrics, exact McNemar's for binary metrics, plus a case-level pass/fail flip list, CLI table and `--json` output | Built and tested |
| Failure clustering | Embeds failing cases, HDBSCAN clusters them, auto-labels each cluster via the judge harness (keyword fallback with no `GROQ_API_KEY`) | Built and tested; `min_cluster_size` default is provisional pending a real-scale golden set |
| `sagwa gate` and CI Action | Threshold-based CI gating from `config/gates.yaml`, plus `--baseline <run_id>` to also fail on a statistically significant regression against a prior run. The reusable GitHub Action (`action.yml`) publishes main's run database as a baseline artifact and gates PRs against it, posting the result as a PR comment | Built and tested; the repo's own workflow still gates the bundled `stub` adapter, not yet a real target |
| Dashboard | Streamlit trend/cost/cluster browser with per-case drill-down (including judge rationale) | Built; query layer is unit-tested, the Streamlit UI itself has not been manually reviewed in a browser |

Summary: the full pipeline described above (ingest, run, metrics, judge, diff, gate, cluster, dashboard) is built and tested end-to-end. Public-benchmark targets and golden sets (HotpotQA, CNN/DailyMail; `benchmarks/`) have been run, and a deliberately degraded HotpotQA pipeline was caught by `sagwa diff` (5 metrics significantly worse, 21 pass→fail flips, 0 the other way). What remains is evidence rather than code: the judge does not yet clear its κ bar, and CI does not yet gate a real target.

## Tech Stack and Architecture

| Layer | Choice |
|---|---|
| CLI | Typer (`sagwa run`, `diff`, `gate`, `cluster`, `dashboard`), installed as a console script |
| Schema validation | Pydantic v2 |
| Storage | SQLAlchemy 2.0 + Alembic, SQLite by default (`sqlite:///./sagwa.db`); Postgres is a config change away, not a rewrite |
| Concurrency | Bounded thread pool (`sagwa/runner/`), no external task queue |
| LLM judge | Groq via `langchain-groq` (optional dependency group `judge`/`metrics`); scores and stores rationale for every case in `sagwa run` |
| RAG metrics | RAGAS, live (optional dependency group `metrics`) |
| Statistical testing | `scipy` (paired bootstrap CI, exact/binomial McNemar's) for `sagwa diff` |
| Clustering | `sentence-transformers` embeddings + HDBSCAN (optional dependency group `clustering`) |
| Dashboard | Streamlit (optional dependency group `dashboard`) |
| Tracing | None yet, deferred; `results.trace_id` is a reserved column for later use |
| CI | GitHub Actions, runs a real `sagwa run` + `sagwa gate` and posts the result as a PR comment |

### Data flow

A golden set is loaded from `golden_sets/*.jsonl` by `load_golden_set()` into a list of `GoldenCase` objects.

Each case is passed to `_load_adapter_class(target)().run(case.input)`, where `target` is either the built-in `stub` adapter or any `module.path:ClassName` resolved dynamically at runtime.

The adapter returns an `AdapterResult`: `answer`, `context`, `latency_ms`, `tokens`, `cost_usd`.

Metrics are computed per case from the `AdapterResult` (`compute_metrics()`, `sagwa/metrics/`): reference-based and classification metrics where ground truth exists, safety flags always, RAGAS for `rag_qa` cases with context, and an LLM-judge score plus rationale whenever `GROQ_API_KEY` is configured. The result, metrics included, is persisted as `Run` and `Result` rows (`sagwa/storage/models.py`) via `get_session()`.

From there, `sagwa diff` (`sagwa/diff/`) compares two runs' `Result` rows; `sagwa gate` (`sagwa/gate/`) evaluates one run's aggregate metrics against `config/gates.yaml`; `sagwa cluster` (`sagwa/clustering/`) groups a run's failing cases; `sagwa dashboard` (`sagwa/dashboard/`) renders trends and the cluster browser. All four only ever read `Result` rows, none of them touch a target pipeline's code.

`sagwa/cli.py` has no hardcoded knowledge of any specific target pipeline, the only built-in adapter is `stub`. Everything else is resolved by importing a `module.path:ClassName` string at runtime, with the caller's working directory added to `sys.path` first (the way `python -m` would resolve it). This is what lets a completely separate project become a target pipeline with zero changes to either codebase.

### Directory guide

- `sagwa/cli.py`: `sagwa run|diff|gate|cluster|dashboard` entry points plus adapter loading
- `sagwa/adapters/`: `TargetAdapter` protocol and `AdapterResult`; stub adapter only, any other target is loaded dynamically
- `sagwa/datasets/`: `GoldenCase` pydantic schema and JSONL loader/validator
- `sagwa/storage/`: SQLAlchemy models (`Run`, `Result`) and session factory
- `sagwa/runner/`: bounded-concurrency case execution
- `sagwa/metrics/`: `reference.py`, `classification.py`, `safety.py`, `judge_metrics.py`, `ragas_metrics.py`, all live
- `sagwa/judge/`: `harness.py` (absolute/pairwise scoring, called from both `sagwa run` and `sagwa/metrics/judge_metrics.py`); `calibration.py`
- `sagwa/diff/`: regression/statistical-significance diff engine (bootstrap CI, McNemar's, per-tag deltas, flip detection)
- `sagwa/gate/`: CI gate; `_predicate.py` holds the shared pass/fail definition `diff` and `clustering` both use, driven by `config/gates.yaml`
- `sagwa/clustering/`: HDBSCAN failure clustering with LLM/keyword auto-labeling
- `sagwa/dashboard/`: `queries.py` (unit-tested query layer) plus `app.py` (Streamlit rendering)
- `sagwa/_embedding.py`: shared lazy `sentence-transformers` model loader, used by both `metrics/reference.py` and `clustering/`
- `examples/adapters/`: reference adapter implementations, kept outside `sagwa/` since they are worked examples, not core library code (includes a README on plugging in any ML project as a target pipeline)
- `golden_sets/`: versioned golden-set JSONL files (`example.jsonl` is a minimal 3-case plumbing check; `demo_synthetic.jsonl` is a 30-case hand-written set spanning all three task types, for a richer worked example)
- `migrations/`: Alembic migrations for the run-history schema
- `calibration/`: judge calibration study artifacts
- `config/gates.yaml`: CI gate thresholds, read by `sagwa gate`/`sagwa diff`/`sagwa cluster` alike

## Prerequisites and Environment Setup

- Python 3.11 or later
- pip (or any PEP 517-compatible installer)
- git (run records pin the current git SHA; required for `sagwa run` to resolve a non-"unknown" `sagwa_git_sha`)
- No database service to run, SQLite ships with the Python standard library

### Environment variables (`.env.example`)

| Variable | Required for | Description |
|---|---|---|
| `GROQ_API_KEY` | LLM-as-judge (`sagwa/judge/`, called from `sagwa run` and `sagwa cluster`'s auto-labeling) and RAGAS (`sagwa/metrics/ragas_metrics.py`) | The only external API this project calls directly. Judge scoring and RAGAS both degrade gracefully (omitted or `None`, not a crash) when unset. |
| `DATABASE_URL` | Storage | Defaults to `sqlite:///./sagwa.db`. Swap for a Postgres DSN to switch stores; the schema is database-agnostic. |

Target-pipeline-specific variables (e.g. a repo path or API key your own adapter needs) are not Sagwa core config; each adapter owns and documents its own env vars. See `examples/adapters/README.md`.

## Quickstart / Installation

```bash
git clone <this-repo-url> sagwa && cd sagwa

# Install with the optional dependency groups you need
pip install -e ".[metrics,judge,clustering,dashboard,dev]"

# Configure environment
cp .env.example .env
# then fill in GROQ_API_KEY if you'll use the judge harness

# Run database migrations - REQUIRED before tests or `sagwa run` on a fresh
# clone. A sagwa.db file existing does NOT mean it's migrated.
alembic upgrade head   # or, from an installed package without alembic.ini: sagwa migrate

# Run the test suite
pytest

# Run the bundled example golden set against the stub adapter
sagwa run --target stub --dataset golden_sets/example.jsonl
```

Gotcha: `sagwa/storage/db.py` happily creates a `sagwa.db` file on first connection, but Alembic still has to run to create the `runs`/`results` tables. On a fresh clone (or after deleting `sagwa.db`), skipping `alembic upgrade head` fails tests with `no such table: runs`.

### Optional dependency groups

| Group | Adds | Used for |
|---|---|---|
| `postgres` | `psycopg[binary]` | Swapping `DATABASE_URL` to Postgres |
| `metrics` | `ragas`, `langchain-community<0.4`, `langchain-groq`, `sentence-transformers`, `scipy` | RAGAS RAG metrics, judge model calls, embedding similarity, `sagwa diff`'s bootstrap/McNemar's tests |
| `clustering` | `sentence-transformers`, `hdbscan`, `scipy` | `sagwa cluster` |
| `dashboard` | `streamlit` | `sagwa dashboard` |
| `dev` | `pytest` | Test suite |

Gate config parsing (`pyyaml`) is a core dependency, not an extra, since `sagwa gate`/`diff`/`cluster` all need it.

## Usage Guide

### `sagwa run`: execute a golden set against a target pipeline

```bash
sagwa run --target <name-or-module:Class> --dataset <path-to-jsonl> [--concurrency N] [--resume <run_id>] [--json out.json]
```

- `--target`: either the built-in `stub` adapter, or `module.path:ClassName` pointing at any class implementing `TargetAdapter`, resolved the same way `python -m` would from your current working directory.
- `--dataset`: path to a golden-set JSONL file (schema: `id`, `input`, `expected_output` or `expected_labels`, `task_type` one of `rag_qa`, `summarization`, `classification`, and `tags`).
- `--concurrency`: max concurrent adapter calls (default `5`).
- `--resume`: continue an interrupted run by id, re-running only the cases it hasn't stored yet.
- `--json`: write the run id and case counts to a file (used by the CI Action).

Each run is persisted (append-only) with its Sagwa git SHA, the target pipeline's git SHA (when the adapter exposes `repo_path`), a dataset content hash, and per-case results including latency, tokens, cost, and computed metrics.

Example, bundled stub adapter:

```bash
sagwa run --target stub --dataset golden_sets/example.jsonl
```

Example, a custom adapter:

```bash
sagwa run --target your_module.path:YourAdapterClass --dataset golden_sets/example.jsonl
```

For a richer worked example spanning all three task types (30 cases, including a few negation/ambiguous/multi-label edge cases), swap in `golden_sets/demo_synthetic.jsonl`.

### Integrating your own application

There's no per-task-type or "one size fits all" adapter. It's one small adapter class per real pipeline you want to evaluate, whether that's a RAG system, a classifier, a summarizer, or anything else that turns text in into text out. The adapter doesn't know or care what task type it's doing, that's a property of each golden-set case, not of the adapter. Three steps, always:

**1. Implement the `TargetAdapter` protocol.** It can live anywhere on your filesystem, it never needs to be inside this repo:

```python
class TargetAdapter(Protocol):
    name: str
    def run(self, case_input: str) -> AdapterResult: ...

@dataclass
class AdapterResult:
    answer: str
    context: str | None      # retrieved/supporting context, if your pipeline has any
    latency_ms: int
    tokens: int | None       # None if your pipeline doesn't track it
    cost_usd: float | None   # None if your pipeline doesn't track it
```

One method: take a string in, return a string (plus whatever metadata you have) out. A minimal real example, wrapping some hypothetical pipeline you already have:

```python
# my_adapters/support_triage_adapter.py
import time
from sagwa.adapters.base import AdapterResult

class SupportTriageAdapter:
    name = "support-triage"

    def __init__(self):
        # Your adapter owns its own configuration, Sagwa's core never
        # needs to know these exist. Read env vars, accept constructor
        # args, load a model, whatever your pipeline needs.
        from my_pipeline import TicketClassifier
        self.classifier = TicketClassifier()

    def run(self, case_input: str) -> AdapterResult:
        start = time.perf_counter()
        result = self.classifier.classify(case_input)   # your actual pipeline call
        latency_ms = int((time.perf_counter() - start) * 1000)

        return AdapterResult(
            answer=result.label,
            context=None,
            latency_ms=latency_ms,
            tokens=result.token_count,   # or None if you don't track it
            cost_usd=None,
        )
```

Two optional duck-typed attributes let Sagwa pick up extra metadata without knowing which adapter it is, set them in `__init__` if they apply:

- `self.repo_path`: pins the target's own git SHA into the run record, for a pipeline that lives in its own separately-versioned repo.
- `self.model_label`: a human-readable model identifier, for a pipeline that routes across models/tiers instead of using one static model id.

If your pipeline is normally called over HTTP but the endpoint drops information your metrics need (like retrieved context), prefer calling the underlying function in-process instead, the same way you'd call any other Python code, rather than going through the lossy wire format.

**2. Write a golden set for that pipeline's actual domain.** A JSONL file, one case per line:

```jsonl
{"id": "t-001", "input": "My invoice shows double charges this month.", "expected_labels": ["billing"], "task_type": "classification", "tags": ["billing"]}
{"id": "t-002", "input": "The app crashes every time I open settings.", "expected_labels": ["bug"], "task_type": "classification", "tags": ["bug"]}
```

`expected_output` (a single reference string) or `expected_labels` (a label set) are both optional per case, whichever fits the ground truth you have. Neither is required: cases without either still get safety flags and an LLM-judge score.

**3. Point Sagwa at it, no registration needed anywhere in Sagwa's own code:**

```bash
sagwa run --target my_adapters.support_triage_adapter:SupportTriageAdapter --dataset golden_sets/support_triage.jsonl
```

The module path is resolved the same way `python -m` would, from your current working directory. From here, `sagwa diff`/`sagwa gate`/`sagwa cluster`/`sagwa dashboard` all work exactly as documented above, unchanged, since they only ever operate on the stored `Result` rows, never on your pipeline's own code.

See `examples/adapters/README.md` for a full worked example against a real external application.

### `sagwa diff`: compare two runs

```bash
sagwa diff --baseline <run_id> --candidate <run_id> [--gates-config config/gates.yaml] [--json out.json]
```

Reports, per metric, the baseline/candidate means and delta, with a paired bootstrap CI for continuous metrics (fuzzy match, ROUGE-L, embedding similarity, judge score, RAGAS scores) or an exact McNemar's test for binary metrics (exact match, safety flags). It also lists cases that flipped pass/fail between runs, grouped deltas per tag, and cases present in only one run. Prints a table and, with `--json`, writes machine-readable output. Pass/fail is defined by `config/gates.yaml`'s thresholds, shared with `gate` and `cluster` below, not a diff-specific config.

### `sagwa gate`: threshold-gate one run

```bash
sagwa gate --run-id <run_id> [--config config/gates.yaml] [--baseline <run_id>] [--json out.json]
```

Reads `config/gates.yaml` (version-controlled, per-metric thresholds), computes each configured metric's mean across the run, and compares it against its threshold. Exits non-zero if any metric fails, including a metric that was configured but never computed for the run (that fails loudly, it does not get silently skipped). Example config:

```yaml
metrics:
  safety.pii.flagged:
    op: lte
    value: 0.0
  safety.toxicity.flagged:
    op: lte
    value: 0.0
  judge.score:
    op: gte
    value: 0.6
  ragas.faithfulness:
    op: gte
    value: 0.85
```

Keys are dotted paths into a `Result`'s `metrics_json` (see the metrics table above for what each metrics module produces), not bare metric names.

With `--baseline <run_id>`, the gate also fails any metric whose change from that run is statistically significant, larger than the config's `regression.tolerance`, and in the bad direction.

This repo's own workflow (`.github/workflows/eval-gate.yml`) uses the action described below, but still points at the bundled `stub` adapter. Note that `judge.score`, `ragas.*` need a `GROQ_API_KEY` repository secret; without it those metrics are never computed and the gate fails them. Wiring in a real target is a deployment-time decision, not a code change.

### `sagwa cluster`: group a run's failures

```bash
sagwa cluster --run-id <run_id> [--gates-config config/gates.yaml] [--min-cluster-size 3] [--json out.json]
```

Embeds every case that fails per `config/gates.yaml` (same pass/fail definition `diff` uses), clusters them with HDBSCAN (density-based, no fixed cluster count), and auto-labels each cluster via the judge harness (falls back to a keyword-frequency label if `GROQ_API_KEY` isn't set). `--min-cluster-size` must be `>= 2`.

### `sagwa dashboard`: browse trends and failures

```bash
sagwa dashboard [--port 8501]
```

Launches a Streamlit app: metric trends and cost/latency trends per target pipeline over time, and a failure-cluster browser with per-case drill-down including the judge's rationale text.

## Use Sagwa in your team's repo

Sagwa ships as a GitHub Action. Point it at your adapter and golden set, and
every PR gets a quality check that compares against your default branch.

```yaml
# .github/workflows/eval.yml
name: eval
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  pull-requests: write
  actions: read          # needed to read the baseline from main's workflow run

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: <owner>/sagwa@v1
        with:
          target: my_pkg.adapters:MyAdapter      # module.path:ClassName
          dataset: golden_sets/my_set.jsonl
          gates-config: config/gates.yaml
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

You need three things in your repo:

1. **An adapter** — one class with `run(case_input) -> AdapterResult`. Your
   pipeline needs no changes; see [examples/adapters/README.md](examples/adapters/README.md).
2. **A golden set** — a JSONL file of `{id, input, expected_output, task_type, tags}`.
3. **A gate config** — thresholds per metric, plus an optional `regression:`
   section (see [config/gates.yaml](config/gates.yaml)).

### How the baseline works

A push to your default branch uploads its run database as the `sagwa-baseline`
artifact. A pull request downloads it and runs `sagwa gate --baseline`, so a
metric fails when its change is **statistically significant and in the bad
direction** — not merely different. Fixed thresholds alone can't express
"worse than main", which is the question a reviewer actually has.

The SQLite file *is* the baseline store: no service to run, nothing to
provision. Teams that want shared, permanent history point `DATABASE_URL` at
Postgres instead and skip the artifacts entirely.

**Before the first baseline exists** — and after GitHub expires artifacts at 90
days — the gate says so and falls back to absolute thresholds rather than
failing mysteriously.

### Cost and runtime, honestly

Per case, the dominant cost is the metrics, not your pipeline. With RAGAS
enabled, a 20-case run takes roughly 45 minutes on a free Groq tier, and a
50-case RAG run consumes about 170K tokens — which is a whole day's budget on
`gpt-oss-120b`'s free tier (200K/day).

Two knobs, both env vars:

- `SAGWA_RAGAS_METRICS=faithfulness` — compute one RAGAS metric instead of two.
  Leave it unset for the full set; set it to `""` for none, which makes a
  50-case run take minutes rather than an hour.
- `SAGWA_RAGAS_MODEL` — score RAGAS with a different model than the judge, so
  the two draw on separate per-model rate limits.

A practical split: a small, fast golden set on every PR, and the full set
nightly or on merges to main.

### Judge trustworthiness

Before gating on `judge.score`, calibrate the judge against human labels
(`sagwa`'s own study, 200 HelpSteer2 human labels: [calibration/calibration_v1-default-rubric_2026-09-20T144450.406736+0000.json](calibration/calibration_v1-default-rubric_2026-09-20T144450.406736+0000.json), κ = 0.450).
`require_calibration()` refuses a judge with no recorded agreement above your
kappa threshold, deliberately. An uncalibrated LLM judge that fails open is
worse than no judge — in Sagwa's own benchmark (`benchmarks/out/diff_hotpotqa.json`) it rated a
measurably worse pipeline 0.22 *higher* than its baseline, while reference metrics and RAGAS faithfulness caught the regression.

## Roadmap and Contributing

### Current status

The full pipeline is built and tested: golden-set schema/loader, run-history storage, adapter contract, the async runner, reference/classification/safety metrics, the judge harness (wired into every run, not just calibration), the calibration engine, `sagwa diff`, `sagwa gate` plus a real CI Action, `sagwa cluster`, and the Streamlit dashboard.

What's left is mostly evidence, not missing implementation:

- The judge's measured κ (0.450) is below the 0.70 target, so `judge.score` should not be the only metric a gate relies on. A revised rubric has not been re-measured.
- The first real regression verdict comes from public benchmarks (HotpotQA with a deliberately degraded v2), not from a product's own traffic.
- The repo's own CI workflow gates the bundled `stub` adapter; a real target needs the `GROQ_API_KEY` secret and a token budget per run.
- `sagwa cluster`'s `min_cluster_size` default is provisional, not tuned at real scale.
- The dashboard's Streamlit UI has not been manually reviewed in a browser (its query layer is unit-tested).
- A free Groq tier allows roughly one 50-case RAG run per model per day, which shapes how `benchmarks/run_benchmarks.sh` is used (it deletes its database first).

### Upcoming

- Re-measure the judge with the revised rubric and report κ against a naive-prompt baseline
- Point the CI Action at a real target and show one blocked and one passing PR
- Measure the cost saving from per-role model routing (`SAGWA_*_MODEL` is built; the saving is not yet quantified), and record an end-to-end demo

Deferred beyond this scope: online/production traffic evaluation, multi-judge ensembling, a hosted multi-tenant version, and other longer-horizon initiatives.

### Contributing

This is currently a solo portfolio/infrastructure project. If you're extending it:

1. Read a module's own docstring before touching it, each was written to point at a specific PRD requirement, not left as a stub by accident.
2. Treat existing architectural choices (SQLite default, deferred tracing) as deliberate rather than bugs to "fix" without discussion.
3. Run `alembic upgrade head` and `pytest` before and after any change.
4. Open an issue or PR describing the change and its motivation.

## License and Acknowledgments

Licensed under the Apache License 2.0. See `LICENSE`.

Sagwa is designed to evaluate any real-world target pipeline purely through the external `TargetAdapter` contract, with no shared code or coupling to that pipeline's internals.
