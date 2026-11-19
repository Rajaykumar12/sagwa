# ADR 0006: the CI baseline is a SQLite file in a GitHub Actions artifact

**Status**: Accepted (2026-09-23)

## Context

`sagwa gate` could only check a run's **absolute** metric values against fixed thresholds. The CI workflow migrated a fresh database on every run, so each PR started from empty history. That left the tool unable to answer the only question a reviewer actually asks — *is this PR worse than main?* — which is PRD G2 and G4 together.

The gap was not analysis: `sagwa diff` already computes paired deltas with bootstrap CIs and exact McNemar's tests, and it demonstrably works (the HotpotQA v1→v2 regression, docs/STATUS.md §2a). What was missing was somewhere for CI to *keep the baseline run* between workflow runs.

Three options were considered:

1. **A hosted database** (Postgres) that every CI run writes to. Correct and permanent, but it obliges every adopting team to provision and secure a database before they can evaluate a single PR.
2. **Re-running the baseline on every PR** — evaluate main and the branch in the same job. No storage at all, but it doubles the API cost and runtime of every PR, and on a free tier one 50-case RAG run already consumes a day's token budget (STATUS.md §3a).
3. **Storing the run database as a CI artifact**, published by the default branch and downloaded by PRs.

## Decision

**Option 3.** A push to the default branch uploads `sagwa.db` plus `baseline_run_id.txt` as the `sagwa-baseline` artifact; a `pull_request` downloads the most recent one and passes it to `sagwa gate --baseline`.

Supporting decisions made alongside it:

- **A metric fails only when its change is statistically significant *and* in the bad direction.** Direction is derived from that metric's existing `op` in `gates.yaml` (`gte`/`gt` = higher is better), so directionality is declared exactly once and cannot disagree with itself. An optional `regression.tolerance` ignores changes too small to care about, because with enough cases a trivial drop can still be significant.
- **Significance comes from `diff_runs()`**, not a second implementation. The gate and `sagwa diff` can therefore never disagree about what regressed.
- **Migrations ship inside the wheel** (`sagwa/_migrations`, via `sagwa migrate`). A consuming repo has no `alembic.ini` and no `migrations/` directory, so `alembic upgrade head` — the documented local command — simply cannot work there.
- **Only the default branch publishes a baseline.** A PR's own run must never become the thing later PRs are measured against.

## Consequences

- **A team adopts Sagwa with a workflow file and a secret.** No database, no infrastructure. That was the deciding factor.
- **Baselines expire.** GitHub keeps artifacts for 90 days, so a quiet repo loses its baseline. The action says "no baseline found" and falls back to absolute-only gating rather than failing in a way that looks like a regression.
- **The baseline is only as good as the last default-branch run.** If that run was itself broken (rate-limited, half-failed), PRs compare against damaged data. `benchmarks/finish_run.sh` already refuses to diff a run with >10% errored cases; the same guard belongs in the action eventually.
- **Postgres remains the answer for shared, permanent history** — point `DATABASE_URL` at it and ignore the artifacts ([ADR-0004](./0004-sqlite-default-defer-postgres.md) keeps the schema database-agnostic precisely for this).
- **Artifacts are readable by anyone who can read the repo's Actions.** The database contains golden-set inputs and pipeline outputs, so a team whose golden set is sensitive should use the Postgres path instead. Worth stating plainly rather than discovering later.
- **MCP is deferred.** It was considered as the "make Sagwa usable elsewhere" step and rejected for now: it would be another way to reach the same `run`/`diff`/`gate` functions, with no named client, while the thing actually blocking adoption was the missing baseline. Revisit once teams use Sagwa in CI.
