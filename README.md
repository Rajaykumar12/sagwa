# Sagwa

**CI for LLM quality.** Run a golden dataset against any prompt, model or pipeline, and fail the PR when quality regresses, the way a failing unit test would.

Python 3.11+ · Typer CLI · SQLAlchemy/Alembic (SQLite by default) · Apache 2.0 · early alpha

## Why

Most teams ship LLM changes on vibes: tweak a prompt, skim five outputs, merge. Quality regresses silently, and there is no record of when or why. Sagwa is the missing test layer:

1. **Golden set**: a versioned JSONL file, reviewed like code.
2. **Run** it against your pipeline through a small adapter and get per-case metrics.
3. **Diff** two runs with statistical significance, not eyeballing.
4. **Gate** CI so a regressing PR fails.
5. **Cluster** the failures to see systemic issues.

## It catches real regressions

On HotpotQA, a plausible cost-cutting change (retrieve 1 paragraph instead of 3, drop the "say I don't know" instruction) was caught by `sagwa diff`:

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| exact match | 0.489 | 0.149 | **−0.340** |
| RAGAS faithfulness | 0.630 | 0.300 | **−0.330** |
| ROUGE-L | 0.549 | 0.329 | **−0.220** |

21 cases flipped pass→fail, none the other way. Evidence is in `benchmarks/out/`.

The same run shows why one metric is not enough: the LLM judge scored the *worse* version **0.22 higher**, rewarding fluent, ungrounded answers. Reference metrics, RAGAS and the judge disagreed in direction, so Sagwa gates on several at once.

## Quickstart

```bash
git clone https://github.com/Rajaykumar12/sagwa && cd sagwa
pip install -e ".[metrics,judge,clustering,dashboard,dev]"
cp .env.example .env          # add GROQ_API_KEY for judge and RAGAS metrics
alembic upgrade head          # required on a fresh clone (or: sagwa migrate)
pytest
sagwa run --target stub --dataset golden_sets/example.jsonl
```

`sagwa.db` existing does not mean it is migrated. Skipping the migration fails with `no such table: runs`. Judge and RAGAS metrics need `GROQ_API_KEY`; without it they are not computed.

## Evaluate your own pipeline

Write one small adapter class per pipeline. It can live anywhere, not in this repo:

```python
import time
from sagwa.adapters.base import AdapterResult

class SupportTriageAdapter:
    name = "support-triage"

    def run(self, case_input: str) -> AdapterResult:
        start = time.perf_counter()
        result = my_pipeline.classify(case_input)      # your real call
        return AdapterResult(
            answer=result.label,
            context=None,                              # retrieved context, if any
            latency_ms=int((time.perf_counter() - start) * 1000),
            tokens=None,                               # None if you don't track it
            cost_usd=None,
        )
```

Optional attributes: `repo_path` pins your pipeline's git SHA into the run, and `model_label` names the model for routed pipelines. See [examples/adapters/README.md](examples/adapters/README.md).

Then write a golden set, one case per line. `task_type` is `rag_qa`, `summarization` or `classification`. `expected_output` or `expected_labels` are optional; cases without either still get safety flags and a judge score.

```jsonl
{"id": "t-001", "input": "My invoice shows double charges.", "expected_labels": ["billing"], "task_type": "classification", "tags": ["billing"]}
```

Run it. The module path resolves like `python -m`, from your working directory:

```bash
sagwa run --target my_adapters.support_triage:SupportTriageAdapter --dataset golden_sets/support_triage.jsonl
```

## Commands

| Command | Purpose |
|---|---|
| `sagwa run --target T --dataset D` | Run a golden set. `--concurrency N`, `--json out.json`, and `--resume <run_id>` to continue an interrupted run, re-running only missing cases |
| `sagwa diff --baseline A --candidate B` | Per-metric and per-tag comparison: paired bootstrap CI for continuous metrics, exact McNemar's for binary ones, whether each change is better or worse, the cases that flipped, and the tag most affected |
| `sagwa gate --run-id R` | Exit non-zero if a metric misses its threshold. `--baseline A` also fails on a significant regression against run A |
| `sagwa cluster --run-id R` | Group failing cases with HDBSCAN and label each cluster |
| `sagwa rootcause --run-id R --baseline B` | For agent runs: group failing cases by cause (retry loop, lost handoff, wrong tool, skipped step, ...) with the evidence from each trajectory |
| `sagwa improve --target T --dataset D` | For agents with a prompt store: diagnose failures, have an LLM revise one prompt, re-run, and keep the revision only if it's a significant improvement. Asks before applying; `--dry-run` only proposes |
| `sagwa dashboard` | Streamlit trends for metrics, cost and latency, with per-case drill-down |
| `sagwa migrate` | Create or upgrade the database schema |

### Gate config

Thresholds live in version-controlled YAML. Keys are dotted paths into a case's metrics. A metric that is configured but was never computed fails the gate rather than being skipped.

```yaml
metrics:
  safety.pii.flagged:    {op: lte, value: 0.0}
  judge.score:           {op: gte, value: 0.6}
  ragas.faithfulness:    {op: gte, value: 0.85}

# Used only with `sagwa gate --baseline`: fail when the change is
# statistically significant, larger than `tolerance`, and in the bad direction.
regression:
  tolerance: 0.02
```

## Use it in your repo's CI

```yaml
# .github/workflows/eval.yml
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  pull-requests: write
  actions: read              # read the baseline from main's run

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Rajaykumar12/sagwa@main
        with:
          target: my_pkg.adapters:MyAdapter
          dataset: golden_sets/my_set.jsonl
          gates-config: config/gates.yaml
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Your repo needs an adapter, a golden set and a gate config. The result is posted as a PR comment and the check fails on a regression.

**Baseline:** a push to your default branch uploads its run database as the `sagwa-baseline` artifact, and each PR gates against it, so the question is "is this PR worse than main?" rather than "does it clear a fixed number?". With no baseline yet, or after GitHub's 90-day artifact expiry, the gate says so and uses absolute thresholds. For shared permanent history, point `DATABASE_URL` at Postgres (`pip install -e ".[postgres]"`) and skip the artifacts.

## Cost and runtime

The metrics dominate the cost, not your pipeline. A 50-case RAG run with RAGAS uses about 170K tokens, which is a full day of `gpt-oss-120b`'s free tier (200K/day) and takes around an hour. Two env vars help:

- `SAGWA_RAGAS_METRICS=faithfulness`: one RAGAS metric instead of two. Set it to `""` for none, which brings a 50-case run down to minutes.
- `SAGWA_JUDGE_MODEL`, `SAGWA_RAGAS_MODEL`, `SAGWA_CLUSTER_MODEL`: route each role to a different model so each draws on its own rate limit. The judge defaults to `gpt-oss-120b`, the model it was calibrated on; moving it to a smaller model saves quota but gives you an unmeasured judge.

A practical split: a small, fast golden set on every PR, the full set nightly or on merges to main.

## Limitations

- **The judge is not trustworthy enough to gate alone.** The default judge (`v3-naive`: a one-line prompt on `gpt-oss-120b`) agrees with 100 held-out human labels at Cohen's κ 0.66, below the 0.70 target, so `require_calibration()` refuses to let it gate. It replaced a detailed rubric that scored 0.32–0.46; see `calibration/calibration_report.md` §7. Gate on the judge alongside reference metrics and RAGAS, never by itself.
- **The regression evidence comes from public benchmarks** (HotpotQA, CNN/DailyMail in `benchmarks/`), not from a product's own traffic.
- **This repo's own CI gates an offline pipeline**: Lead-3 summarization on CNN/DailyMail (`benchmarks/offline.py`), on reference and safety metrics only (`config/gates.ci.yaml`), so it needs no API key. A one-line Lead-3→Lead-1 PR passed every absolute threshold and was blocked by the baseline comparison ([#2](https://github.com/Rajaykumar12/sagwa/pull/2)); a no-op PR passed ([#1](https://github.com/Rajaykumar12/sagwa/pull/1)). CI does not gate an LLM target, because that needs a key and a token budget per PR.
- **Clustering and dashboard are provisional:** `min_cluster_size` is untuned at real scale. The dashboard works on real runs but groups trends by adapter class and always judges failures by `config/gates.yaml`.
- **Free API tiers limit throughput** to roughly one 50-case RAG run per model per day. `benchmarks/run_benchmarks.sh` deletes its database first.
- **Tracing is deferred:** `results.trace_id` is a reserved column.

Storage is SQLite by default; the schema is database-agnostic, so Postgres is a config change.

## License

Apache 2.0. See `LICENSE`.
