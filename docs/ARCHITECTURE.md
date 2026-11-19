

# ARCHITECTURE.md

Describes the system **as it actually exists today** (last updated 2026-09-22, when the
pipeline produced its first real regression verdict end to end: golden set → run →
metrics → gate → diff → clusters, against HotpotQA RAG-QA and CNN/DailyMail
summarization targets built from public benchmarks in `benchmarks/`. Five defects that
only real runs could surface were fixed along the way — the judge silently returning no
score, NaN poisoning gate aggregates, NaN voiding a paired diff, LLM clients hanging
forever with no request timeout, and clustering crashing below `min_cluster_size`. Originally written 2026-08-20, Milestones 1-6 of the then-current gap analysis (now [STATUS.md](./STATUS.md)) implemented — diff, gate + CI Action, clustering, and the dashboard joined the already-built run/metrics/judge pipeline, the judge harness is wired into `sagwa run` itself, the RAGAS dependency blocker is resolved, and classification metrics against `GoldenCase.expected_labels` are now wired in too, closing a gap that existed since the schema was first written). For product intent and full v1 scope, see [PRD.md](./PRD.md); for what is and isn't finished, see [STATUS.md](./STATUS.md). Tech-stack rationale is §3 below. Deviations from the original plan are recorded as ADRs in `docs/adr/` and noted inline.

---

## 1. Status: current vs. planned

| Layer | PRD requirement | Status |
|---|---|---|
| Golden-set schema/loader | FR-1, FR-2, FR-3 | ✅ Built, tested (`sagwa/datasets/`) |
| Run-history storage | FR-6, FR-7, FR-7a | ✅ Built, tested (`sagwa/storage/`); `results.error` added (migration 0002) for per-case failure capture |
| Target-adapter contract | FR-3a | ✅ Built (`sagwa/adapters/base.py`) — and, as of this pass, genuinely pluggable: `sagwa/cli.py` has no hardcoded knowledge of any specific target pipeline (see §2) |
| Stub adapter | — (plumbing validation) | ✅ Built — the one zero-config, always-available target |
| Real target pipelines | FR-3b, FR-3c | ✅ `benchmarks/adapters.py`: `HotpotV1`/`HotpotV2` (RAG-QA over 492 pooled HotpotQA paragraphs, dense retrieval via the shared `sagwa/_embedding.py` model) and `CnnDmSummarizer` (FR-3c's second task type). **All three have now run end to end for real** (2026-09-20/22): v1 (50 cases), v2 (50 cases, 0 errors) and the summarizer (30 cases, gate passed). The v1→v2 diff caught a real regression — see [STATUS.md §2a](./STATUS.md). Retrieval choice was measured, not assumed: at top_k=3, dense retrieval finds both supporting paragraphs for 32/50 questions vs BM25's 23 and TF-IDF's 4 |
| Async/concurrent runner | FR-5, FR-6 | ✅ Built, tested (`sagwa/runner/`) — bounded thread pool, configurable `--concurrency`; a failing case is captured, not fatal to the run |
| `sagwa run` CLI | FR-4 | ✅ Wired end-to-end against `stub` and any `module.path:ClassName` adapter; computes metrics per case; pins `target_pipeline_git_sha`/`model` when the adapter exposes optional `repo_path`/`model_label` attributes |
| Metrics: reference-based, safety | FR-8, FR-12 | ✅ Built, tested (`sagwa/metrics/reference.py`, `safety.py`) — exact/fuzzy/ROUGE-L/embedding-similarity; PII regex + toxicity keyword-list flags (documented as a v1 heuristic, not ML-grade) |
| Metrics: classification | FR-1, FR-8 | ✅ Built, tested (`sagwa/metrics/classification.py`) — set-based precision/recall/F1 + exact-set-match against `GoldenCase.expected_labels`, which the schema always declared but `compute_metrics()` never previously read (only `expected_output` was wired in). Adapter output is parsed as a comma/semicolon/newline-separated label set, so both single- and multi-label predictions work through one path. |
| Metrics: RAGAS (faithfulness, context precision) | FR-9 | ✅ Built and functional (`sagwa/metrics/ragas_metrics.py`) — as of 2026-09-20 the scoring model and metric subset are configurable (`SAGWA_RAGAS_MODEL`, `SAGWA_RAGAS_METRICS`, see §6) and the client has a 90s timeout. **Known limitation, measured on real data**: faithfulness scores a correct refusal as 0 or NaN (11 of 47 HotpotQA v1 cases), so it cannot distinguish an honest "I don't know" from a hallucination — gating on it alone punishes a cautious pipeline. Separately, the `ragas==0.4.3` vs `langchain-community` import blocker is fixed (pinned `langchain-community<0.4` in the `metrics` extra; `ragas` declared no version bound and an unpinned install grabbed a release that had dropped the module `ragas` imports). A second bug the fix exposed — `ChatGroq` rejecting `reasoning_effort` inside `model_kwargs` — is also fixed. Live-verified against Groq (2026-08-19): real scores returned. Still degrades to `None` per metric on any other failure (e.g. no `GROQ_API_KEY`) rather than crashing the run. |
| LLM-as-judge harness (absolute + pairwise) | FR-10, FR-11 | ✅ Built, tested (`sagwa/judge/harness.py`); `groq_llm_call()` verified **live** against the real Groq API. **Fixed 2026-09-20**: `max_tokens` was 60, and gpt-oss models spend their budget on hidden reasoning before emitting `content` — 4 of 10 real calibration cases came back with no score at all, silently. Now 512 (0/10 empty). Also: the client had no request timeout, so a dropped connection hung a run indefinitely rather than erroring; now 90s. Default model is `openai/gpt-oss-20b` with `reasoning_effort="low"`. |
| Judge calibration engine | FR-13, FR-14, FR-15, FR-15a | ✅ Built, tested (`sagwa/judge/calibration.py`) — Cohen's kappa, confusion matrix, versioned JSON artifacts, `require_calibration()` gate-refusal, and `baseline_name` for the FR-15a comparison (now against a naive judge prompt — see [ADR-0005](./adr/0005-public-benchmarks-over-ringo.md)). **Run for real against 200 human labels** (HelpSteer2, 2026-09-20): **κ = 0.450**, accuracy 72.5%, 0 unscoreable cases — below the FR-15 bar of 0.70, so `require_calibration()` refuses this judge for gating. The error is one-sided (too generous) and the judge's *ordering* is sound. **Re-measured on a held-out half (2026-11-19)**: prompt v2 (`v2-banded-rubric`) 0.32 on 20b and 0.44 on 120b; the FR-15a naive judge 0.66 on 120b, beating the rubric by +0.22 [+0.10, +0.36]. No judge clears 0.70. **The default is now `v3-naive`** (that one-line prompt on `gpt-oss-120b`, κ 0.66), still showing the expected answer and context when present; v2 is kept as `BANDED_RUBRIC_V2`. `benchmarks/calibrate.py` takes `--model` and `--holdout-only`. See [calibration/calibration_report.md](../calibration/calibration_report.md) §7. |
| Judge wired into `sagwa run` | FR-10, FR-28 (prereq) | ✅ Built, tested (`sagwa/metrics/judge_metrics.py`) — `compute_metrics()` now calls `score_absolute_with_rationale()` per case when `GROQ_API_KEY` is set, storing both score and raw rationale under `Result.metrics_json["judge"]`; omits the key (not a crash) when no key is configured. Verified live against Groq. |
| `sagwa diff` | FR-16–FR-19 | ✅ Built, tested (`sagwa/diff/`), and **validated on a real regression** (5 metrics significantly worse, 21 pass→fail flips, per-tag split isolating multi-hop damage — [STATUS.md §2a](./STATUS.md)) — per-metric aggregate deltas (paired bootstrap CI for continuous metrics, exact/binomial McNemar's for binary metrics), per-tag grouping, case-level pass/fail flip detection, CLI table + `--json` output. **Fixed 2026-09-22**: `_metric_deltas` treated NaN as a value, so one unscoreable case voided an entire paired comparison — it reported `ragas.faithfulness … nan` on the run where faithfulness showed the second-largest regression. `_is_scored()` now drops NaN pairs, mirroring `sagwa.gate`. Pass/fail is defined against `config/gates.yaml` thresholds (shared with gate/clustering, see `sagwa.gate._predicate.case_passes`). |
| Failure clustering | FR-20–FR-22 | ✅ Built, tested (`sagwa/clustering/`) — HDBSCAN over embedded failing-case text (shared embedding loader, `sagwa/_embedding.py`), sorted by cluster size; auto-labels via the judge harness's `groq_llm_call()` (capped to the top-10 clusters by size) with a keyword-frequency fallback when no `GROQ_API_KEY` is set. `min_cluster_size=3` held up unchanged on 49 real failing cases (two coherent clusters + outliers, [STATUS.md §2a](./STATUS.md)), though it is validated at only one scale. Returns `[]` rather than raising when a run has fewer failing cases than `min_cluster_size` — hdbscan's kd-tree errors there. |
| `sagwa gate` + CI Action | FR-23–FR-25 | ✅ Built, tested (`sagwa/gate/`) — reads `config/gates.yaml`, aggregates (mean) each configured metric across a run's cases, per-op comparison (`gt/gte/lt/lte`); a configured-but-uncomputed metric fails loudly (`observed=None`) rather than being skipped. **Fixed 2026-09-20**: NaN is now read as "couldn't be scored", not as a value. RAGAS returns NaN for a case with no claims to check (a correct "I don't know"), and a single NaN among 47 cases turned the run's mean into `nan` and failed the metric — a real gate verdict that looked like a measurement. `aggregate_metric` and `_predicate.case_passes` both skip it. Non-zero exit on failure. `sagwa gate --baseline` also fails a metric on a statistically significant regression against a baseline run ([ADR-0006](./adr/0006-ci-baseline-via-artifact.md)). `action.yml` at the repo root is the reusable action; `.github/workflows/eval-gate.yml` dogfoods it on a real pipeline with no API key: Lead-3 extractive summarization over the CNN/DailyMail golden set (`benchmarks/offline.py`, gated by `config/gates.ci.yaml`). **Exercised on real PRs (2026-11-19)**: a no-op PR passed ([#1](https://github.com/Rajaykumar12/sagwa/pull/1)); a Lead-3→Lead-1 PR passed every absolute threshold and was failed on the baseline comparison ([#2](https://github.com/Rajaykumar12/sagwa/pull/2)). That demo also surfaced a bug: the baseline download had been failing silently on every PR (fixed in `bc9c684`). |
| Dashboard | FR-26–FR-28 | ✅ Built — `sagwa/dashboard/queries.py` (unit-tested: metric/cost/latency trends, cluster browsing, case drill-down with judge rationale) + `sagwa/dashboard/app.py` (thin Streamlit rendering layer, launched via `sagwa dashboard`). **Reviewed in a browser against the real benchmark runs (2026-11-19)**: trends, cost/latency and the cluster drill-down all render with 0 console errors (screenshots in `docs/img/`). Known limitations from that review: trends group runs by adapter class, so a v1→v2 change made as two classes never shares a line; failing cases are always judged by `config/gates.yaml`, not the run's own gate config; the "rationale" is the judge's raw completion, which is often just the number; and the cost chart is empty when adapters don't report cost. |

---

## 2. High-level architecture

```
golden_sets/*.jsonl  →  load_golden_set()  →  [GoldenCase, ...]
                                                     │
                                                     ▼
                                  _load_adapter_class(target)().run(case.input)
                                  (sagwa/cli.py — "stub" built in, or ANY
                                   "module.path:ClassName" resolved dynamically)
                                                     │
                                                     ▼
                                              AdapterResult
                                    {answer, context, latency_ms, tokens, cost_usd}
                                                     │
                                                     ▼
                                     Run / Result rows (sagwa/storage/models.py)
                                              persisted via get_session()
```

Everything above is driven by `sagwa run --target <name-or-module:Class> --dataset <path>` (`sagwa/cli.py`). `sagwa diff`, `sagwa gate`, `sagwa cluster` and `sagwa dashboard` are all real commands over the same `Run`/`Result` rows — see §1.

**Adapter loading is deliberately not a hardcoded registry.** `sagwa/cli.py`'s `BUILTIN_ADAPTERS` dict holds only `"stub"` — the one zero-config, always-available target. Anything else is resolved by `_load_adapter_class()`: a `"module.path:ClassName"` string is `importlib.import_module()`-ed and the class pulled off it via `getattr`, with the caller's cwd added to `sys.path` first so a project-relative module path resolves the way `python -m` would. This means integrating a new ML project as a target pipeline never requires editing anything inside `sagwa/` — see `examples/adapters/README.md`. Two optional duck-typed attributes let an adapter report extra run metadata without `cli.py` needing to know which adapter it is: `repo_path` (git SHA pinning, FR-7a) and `model_label` (for a target with no single static model id, e.g. one that routes per-query across model tiers).

## 3. Tech stack (as implemented, not as originally planned)

| Layer | Choice | Why this, over the alternatives |
|---|---|---|
| CLI | Typer | `sagwa run\|diff\|gate\|cluster\|dashboard`, entry point in `pyproject.toml` |
| Schema validation | Pydantic v2 | `GoldenCase` (`sagwa/datasets/schema.py`) — the golden-set schema is the contract everything else builds on, so it validates at the edge and fails with a readable error (FR-2) |
| Eval metrics | **RAGAS** for RAG-specific metrics + a **custom judge harness** | RAGAS covers well-studied metrics (faithfulness, context precision) that there is no reason to reinvent. But it covers none of the orchestration this project is actually about — CI gating, run history, significance testing, clustering — so a thin custom layer wraps it. Building everything from scratch would waste time on solved problems; using only an off-the-shelf eval SaaS would remove the systems design entirely. The middle path is deliberate |
| Runner | `concurrent.futures` thread pool with bounded concurrency | Golden sets run against rate-limited APIs, so concurrency control is the requirement (FR-5), not distribution. Right-sized: no Celery/Kafka for a workload of 50 HTTP calls |
| Storage | SQLAlchemy 2.0 + Alembic, **SQLite by default** | Run history is inherently relational (runs → results, joined for diffing). Postgres + pgvector is the intended production store once clustering wants to persist embeddings; SQLite is the interim default because it needs no running service. The schema is DB-agnostic, so switching is a config change — see [ADR-0004](./adr/0004-sqlite-default-defer-postgres.md) |
| Clustering | `sentence-transformers` + HDBSCAN | HDBSCAN doesn't need the cluster count up front (unlike k-means) — correct here, because you don't know in advance how many failure modes exist |
| Stats | `scipy.stats` — bootstrap resampling, exact McNemar's | Recognized methods, not an ad-hoc "3% is a regression". This is what makes the gating claim survive scrutiny |
| CI | GitHub Actions + `pip install -e .` | Makes the gate concrete: a real PR, a real failing check, a real posted comment |
| Dashboard | Streamlit | Days, not weeks, for trend views — the budget belongs in the eval logic and the calibration study, which are the parts that demonstrate judgment. A Next.js rebuild is a stretch goal, not a capability gap |
| Tracing | None yet | Langfuse deferred; `results.trace_id` exists and takes a trace ID later without a schema change. See [ADR-0002](./adr/0002-defer-langfuse.md) |

RAGAS (see §1), judge (Groq via `langchain-groq`), clustering (`sentence-transformers` + HDBSCAN), and dashboard (Streamlit) optional dependency groups are all now live and exercised by `sagwa/metrics/ragas_metrics.py`, `sagwa/metrics/judge_metrics.py`, `sagwa/clustering/`, and `sagwa/dashboard/` respectively. `scipy` (bootstrap CI / McNemar's, `sagwa/diff/`) moved from the `clustering` extra into `metrics`, so `pip install -e ".[metrics]"` alone is enough for `sagwa diff` — it has no reason to require `hdbscan`. `pyyaml` (gate config parsing) is a core dependency, not an extra, since `sagwa gate`/`sagwa diff`/`sagwa cluster` all need it for the shared `config/gates.yaml` pass/fail definition.

## 4. Directory guide

```
sagwa/
├── cli.py            # `sagwa run|diff|gate` entry points + `_load_adapter_class` (the pluggability mechanism)
├── adapters/          # TargetAdapter protocol + AdapterResult; stub only — no other pipeline baked in
├── datasets/           # GoldenCase pydantic schema + JSONL loader/validator
├── storage/            # SQLAlchemy models (Run, Result) + session factory (reads DATABASE_URL)
├── runner/             # bounded-concurrency case execution (sagwa/runner/__init__.py::run_cases)
├── metrics/            # reference.py, classification.py, safety.py, judge_metrics.py, ragas_metrics.py — all live
├── judge/              # harness.py (absolute/pairwise scoring); calibration.py (kappa, artifacts, gate refusal)
├── diff/               # diff engine: bootstrap CI / McNemar's, per-tag deltas, flip detection (FR-16..19)
├── gate/                # CI gate (FR-23..25); _predicate.py holds the shared pass/fail definition diff/clustering also use
├── clustering/          # HDBSCAN failure clustering + LLM/keyword auto-labeling (FR-20..22)
├── dashboard/            # queries.py (testable) + app.py (Streamlit) — trends, cluster browser, case drill-down (FR-26..28)
└── _embedding.py         # shared lazy sentence-transformers loader (metrics/reference.py and clustering both use it)
examples/adapters/       # reference adapter implementations, deliberately outside sagwa/ (see its README)
└── README.md              # how to plug in any ML project as a target pipeline
golden_sets/            # versioned golden-set JSONL files: 50-case hotpotqa.jsonl and 30-case cnndm.jsonl (real, from public benchmarks), plus the 3-case example.jsonl and 30-case synthetic demo_synthetic.jsonl fixtures
benchmarks/             # the public-benchmark targets: build.py (regenerates every golden set + calibration label file from Hugging Face, fixed seed), adapters.py (HotpotV1/V2, CnnDmSummarizer), calibrate.py, per-benchmark gate configs, run_benchmarks.sh
migrations/              # Alembic migrations for the run-history schema (0001 initial, 0002 adds results.error)
calibration/             # judge calibration: 200 HelpSteer2 human labels, per-judge-version score caches, versioned κ artifacts, calibration_report.md
config/gates.yaml        # default CI gate thresholds (FR-25), read by `sagwa gate`, `sagwa diff` and `sagwa cluster`; benchmarks/ ships its own per-target configs
docs/                    # PRD.md (what & why), ARCHITECTURE.md (this file), STATUS.md (the tracker), AGENTIC_PRD.md (future scope), adr/ (decision log)
```

## 5. Key contracts

**Golden-set row** (`sagwa/datasets/schema.py`, `GoldenCase`): `id, input, expected_output?, expected_labels?, task_type ∈ {rag_qa, summarization, classification}, tags[]`. `id` must be stable across dataset revisions — it's the join key for future diffing (FR-17).

**Adapter contract** (`sagwa/adapters/base.py`): any target pipeline implements `TargetAdapter.run(case_input: str) -> AdapterResult`, where `AdapterResult = {answer, context, latency_ms, tokens, cost_usd}`. Zero required changes to the target's own codebase (FR-3a) — and, symmetrically, zero required changes to Sagwa's own codebase either, via the `module.path:ClassName` loading mechanism (§2).

**Run/Result schema** (`sagwa/storage/models.py`): `Run` pins `sagwa_git_sha`, `target_pipeline_git_sha` (nullable — populated once an external adapter reports it), `dataset_sha256`, `model`, `status`. `Result` rows are per-case, carrying `metrics_json` (empty dict until the metrics layer exists) and `trace_id` (nullable). Both tables are append-only — nothing is ever updated in place after a run completes.

## 6. Environment variables

See `.env.example`. Only two Sagwa-*core* vars: `GROQ_API_KEY` (judge, live-verified) and `DATABASE_URL` (defaults to `sqlite:///./sagwa.db`). Target-pipeline-specific vars are owned by each adapter, not Sagwa's core — see `examples/adapters/README.md`.

Two optional vars exist for one reason: Groq's free tier rate-limits **per model**, so a run that scores everything with one model is throttled by that model's tokens-per-minute budget alone. `SAGWA_RAGAS_MODEL` moves RAGAS onto its own model, and `SAGWA_RAGAS_METRICS` (comma-separated) narrows which RAGAS metrics are computed — a gate config that never reads `context_precision` can skip it and its LLM calls. Both default to the previous behavior. `benchmarks/run_benchmarks.sh` sets both.

## 7. Build / test commands

```bash
pip install -e ".[metrics,judge,clustering,dashboard,dev]"
cp .env.example .env
alembic upgrade head     # REQUIRED before running tests or `sagwa run` on a fresh clone — see gotcha below
pytest
sagwa run --target stub --dataset golden_sets/example.jsonl
```

### Gotcha: migrations must run before tests
`sagwa/storage/db.py` will happily create a `sagwa.db` file on first connection, but Alembic still needs to run to create the actual `runs`/`results` tables. A `sagwa.db` file existing is not the same as it being migrated — on a fresh clone (or after deleting `sagwa.db`), `test_storage_roundtrip.py` fails with `no such table: runs` until `alembic upgrade head` runs. There's no CI/fixture step that does this automatically yet.
