# Product Requirements Document: Sagwa

**LLM Evaluation & Regression-Testing Platform**

| | |
|---|---|
| **Author** | Rajay Kumar |
| **Status** | v1.1 — adds §5.10, agent evaluation |
| **Last updated** | 2026-11-25 |
| **Related docs** | [ARCHITECTURE.md](./ARCHITECTURE.md) — how it works today & tech-stack rationale · [STATUS.md](./STATUS.md) — what's done, what's not |

---

## 1. Summary

Sagwa is CI/CD infrastructure for LLM quality. It lets an engineer define a golden test set for an LLM-powered task, run it against any prompt/model/pipeline version, and get back structured, statistically defensible metrics — with the ability to gate a merge in CI the same way a failing unit test would. It exists to replace "vibes-based" LLM deployment with a testable, auditable process.

This PRD defines *what* is being built and *why it matters to a user*, in product terms. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the technical architecture and tech-stack rationale, and [STATUS.md](./STATUS.md) for current status and remaining gaps.

---

## 2. Problem Statement

Teams shipping LLM features have no equivalent of a test suite. A prompt tweak, a model swap, or a RAG config change ships based on a handful of manually-eyeballed outputs. There is no historical record of quality over time, no automated way to catch a regression before a customer does, and no shared, defensible definition of "good enough" that a team can rally around.

This is not a hypothetical problem — it is the single most common failure mode teams hit within 6–12 months of shipping their first LLM feature: quality drifts silently, an unrelated prompt change breaks an edge case nobody was testing for, and the team only finds out from a support ticket or a bad review.

### Who feels this pain
- **AI/ML engineers** iterating on prompts or RAG pipelines who have no fast, trustworthy signal on "did I make this better or worse"
- **Eng leads / tech leads** who need to approve a PR touching an LLM feature and currently have no artifact to review besides a diff of prompt text
- **Applied AI teams at product companies** scaling from one LLM feature to several, where manual spot-checking stops being tractable

---

## 3. Goals

| # | Goal |
|---|---|
| G1 | Let an engineer run a golden dataset against a target LLM pipeline and receive structured, per-case quality metrics in one command |
| G2 | Let an engineer diff two runs (e.g., before/after a prompt change) and see which specific cases regressed, with a statistically grounded significance call — not just "the average went down" |
| G3 | Automatically group failing cases into semantically coherent clusters so a human can spot systemic issues without reading every transcript |
| G4 | Gate a CI pipeline so a PR that regresses quality below a defined threshold fails the build automatically |
| G5 | Provide a dashboard showing quality, cost, and latency trends across runs over time |
| G6 | Demonstrate — with a documented calibration study — that the automated judge is actually trustworthy, not just plausible |

### Non-goals (explicitly out of scope for v1)
- Building a general-purpose eval SaaS product with multi-tenant auth, billing, etc. — this is a single-user/single-team tool, not a commercial platform
- Supporting every possible LLM task type — v1 covers RAG-QA, summarization, and classification only; other task types are a stated extension point, not a v1 deliverable
- Real-time/online evaluation of live production traffic — v1 is offline/batch evaluation against golden sets; online eval is a documented future direction
- Building or maintaining target pipelines to production quality as part of *this* project — a target pipeline is something Sagwa evaluates through a stable adapter contract, never something it owns. The v1 targets are built from public benchmarks (see §4) precisely so that Sagwa takes on no responsibility for their quality or roadmap

---

## 4. Users & Use Cases

### Primary persona: **Dana, AI Engineer**
Owns a RAG-based support-doc assistant at a mid-size SaaS company. Wants to try a new retrieval strategy but is worried it'll quietly make answers less faithful to source docs. Currently has no way to check this except manually reading 20 outputs and guessing.

**Use case**: Dana runs `sagwa run --target rag_v2 --dataset golden/support_qa.jsonl`, gets faithfulness/context-precision scores, then runs `sagwa diff --baseline rag_v1 --candidate rag_v2` and sees a table of exactly which questions got worse, each with a judge-generated explanation.

**Concrete v1 instantiation**: this persona is realized against real, messy data rather than a toy fixture. The v1 targets are built from public benchmarks (`benchmarks/adapters.py`): a multi-hop RAG-QA pipeline over pooled HotpotQA paragraphs, shipped as a **baseline and a deliberately-regressed candidate** so the diff and gate have a genuine regression to catch, plus a CNN/DailyMail summarizer proving the machinery generalizes across task types. Judge calibration runs against 200 real human ratings from HelpSteer2.

*History*: through 2026-09-13 the primary target was **ringo**, a separately-maintained RAG chat app, chosen because it shipped an uncalibrated ad-hoc judge worth beating. That adapter was removed (commit `58607f2`) — see [ADR-0005](./adr/0005-public-benchmarks-over-ringo.md) for why, and for what replaced the head-to-head comparison it was meant to provide.

### Secondary persona: **Sam, Tech Lead**
Reviews a teammate's PR that changes a system prompt. Wants an objective artifact to review beyond "the text looks reasonable to me."

**Use case**: The PR's CI check runs Sagwa automatically, posts a comment with a metric diff table, and blocks merge if faithfulness drops below the configured threshold. Sam approves based on the report, not a manual read-through.

### Tertiary persona: **(You) the project author, in an interview**
Needs to walk a hiring manager through a coherent, defensible story about production LLM quality ownership within 5–10 minutes, backed by real artifacts (a calibration report, a CI run, a dashboard).

---

## 5. Functional Requirements

### 5.1 Golden Dataset Management
- FR-1: Golden sets are defined as versioned JSONL/YAML files with a fixed schema (`input`, `expected_output` or `expected_labels`, `task_type`, `tags`)
- FR-2: The system validates a golden set file against its schema and rejects malformed entries with a clear error
- FR-3: Golden sets support free-form `tags` per case (e.g., `multi-hop`, `negation`) to enable slice-based reporting later

### 5.2 Target Pipeline Adapters
- FR-3a: A target pipeline is integrated via a stable adapter interface — `(case_input) -> {answer, context, latency_ms, tokens, cost}` — with zero required changes to the target's own codebase
- FR-3b: The reference v1 adapters are real, runnable pipelines built from public benchmarks (`benchmarks/adapters.py`): `HotpotV1`/`HotpotV2` (RAG-QA over a pooled 492-paragraph HotpotQA corpus, dense retrieval) and `CnnDmSummarizer`. Sagwa's own code contains no knowledge of any of them — they load via `module.path:ClassName` like any third-party target. *Supersedes the original ringo-based requirement; see [ADR-0005](./adr/0005-public-benchmarks-over-ringo.md).*
- FR-3c: A second adapter (public-dataset summarization task) exists solely to prove the adapter interface generalizes across task types without storage/CLI changes

### 5.3 Eval Execution
- FR-4: A user can run a named golden set against a named target pipeline via CLI: `sagwa run --target <name> --dataset <path>`
- FR-5: Execution is concurrent/batched with configurable rate limiting to respect API limits
- FR-6: Each run captures, per case: raw output, latency, token usage, estimated cost, and a trace ID
- FR-7: A run is uniquely identified and persisted with its git SHA, prompt version, model identifier, and dataset version
- FR-7a: Where the target pipeline is an external, independently-versioned system, the run record additionally pins the **target pipeline's own git SHA** — not just Sagwa's — since the two repos version independently and a regression could originate on either side

### 5.4 Metrics
- FR-8: The system computes reference-based metrics (exact/fuzzy match, ROUGE, embedding similarity) where ground truth exists
- FR-9: The system computes reference-free RAG metrics (faithfulness, context precision, context recall) via RAGAS for RAG-type tasks
- FR-10: The system computes LLM-as-judge rubric scores for tasks without clean ground truth (e.g., open-ended generation quality)
- FR-11: The judge supports both absolute scoring and pairwise comparison modes
- FR-12: The system flags safety issues (PII leakage, toxicity) per case as a boolean/severity flag, separate from quality metrics

### 5.5 Judge Calibration
- FR-13: The system provides a calibration workflow: given a set of judge scores and corresponding human labels, compute agreement (accuracy, Cohen's κ) and output a confusion matrix
- FR-14: Calibration results are stored as a versioned artifact tied to a specific judge prompt version, so a judge-prompt change can be re-validated
- FR-15: The system must refuse (or at minimum, loudly warn) to use a judge for gating decisions if it has no recorded calibration result above a configurable κ threshold
- FR-15a: The calibration report supports an explicit **baseline comparison mode** — scoring a second, weaker judge against the same human-labeled set, so Sagwa's judge is reported against a real prior baseline rather than an absolute κ alone. The v1 baseline is a **naive one-line judge prompt** (no rubric, no scale definition, no reasoning budget) standing in for the ad-hoc "single uncalibrated LLM call" that teams actually ship. *Originally specified as a head-to-head against ringo's `backend/eval.py`; see [ADR-0005](./adr/0005-public-benchmarks-over-ringo.md) for why that baseline is gone and why this one tests the same claim — that calibration, not model choice, is what makes a judge trustworthy.*

### 5.6 Regression Detection (Diff)
- FR-16: A user can diff two runs: `sagwa diff --baseline <run_id> --candidate <run_id>`
- FR-17: The diff reports per-metric, per-tag aggregate deltas, plus a list of individual cases that flipped pass→fail or fail→pass
- FR-18: Aggregate metric deltas are accompanied by a statistical significance test (bootstrap CI or McNemar's, as appropriate to the metric type) — not reported as a bare percentage
- FR-19: The diff output is available as both a CLI table and a machine-readable JSON (for CI parsing)

### 5.7 Failure Clustering
- FR-20: Failing cases from a run are embedded and clustered (density-based, no fixed cluster count required)
- FR-21: Each cluster receives an auto-generated natural-language label summarizing the common failure pattern
- FR-22: Cluster results are browsable in the dashboard, sorted by cluster size

### 5.8 CI Gating
- FR-23: A GitHub Action wraps the CLI and runs on PR, with pass/fail thresholds configured per metric (e.g., `faithfulness>=0.85`)
- FR-24: On failure, the Action posts the diff summary as a PR comment and fails the check
- FR-25: Gate thresholds are defined in a version-controlled config file, not hardcoded

### 5.9 Dashboard
- FR-26: A dashboard displays metric trends over time (per target pipeline)
- FR-27: A dashboard displays cost and latency trends over time
- FR-28: A dashboard allows browsing a single run's failure clusters and drilling into individual cases with judge rationale

### 5.10 Agent evaluation

An agent's final answer is a lossy summary of what it did: a planner change can leave the answer looking fine while doubling tool calls, looping on a failing tool, or dropping context between agents. These requirements evaluate the trajectory, not only the output. Numbering continues from §5.9 so modules can cite FR-NN.

- FR-29: An agentic target returns its **trajectory** with its answer: an ordered list of agent turns, tool calls and handoffs, each with input, output, latency, tokens, cost and any error. Existing (non-agent) adapters work unchanged.
- FR-30: Trajectories are stored **one row per step**, linked to the case's result, so metrics and the trace viewer query steps directly. A run with no trajectory stores none.
- FR-31: A golden case can be **agentic**: it must declare **success criteria** (answer patterns, tools that must or must not be used, a step limit, whether step errors are allowed, whether the right answer is to decline) and may list the tools a good trajectory uses.
- FR-32: **Task success** is decided deterministically from those criteria, never by the LLM judge and never from the gate config, and the conditions a case missed are recorded.
- FR-33: Per-case **trajectory metrics**: tool-selection accuracy, tool error, retry and loop rates, recovery after a failed call, handoff context loss, steps, tokens, latency and cost. Run-level ratios (tokens or cost per successful task) are computed from run totals, not averaged per case.
- FR-34: Every metric is declared once, with its kind and good direction, and the diff and gate read that declaration. A computed metric that isn't declared is a test failure, not a silent gap.
- FR-35: The diff reports whether each change is **better or worse**, reports run-level metrics without a paired significance test, and names the **most affected tag**. An agent gate profile sets thresholds on success and trajectory metrics.
- FR-36: A **trace viewer** walks run → cases (filterable to failures) → one case's trajectory, with failed steps highlighted, a per-agent / per-tool roll-up, and any step's inputs, outputs, tool arguments and model.
- FR-37: Failure clustering groups agentic failures by **trajectory signature** (what the agent did) rather than by question and answer text.
- FR-38: **Root-cause analysis** assigns each failing agentic case one cause from a closed taxonomy (retry loop, step budget exhausted, scope, handoff loss, tool selection, skipped step, grounding, unknown) with evidence and a confidence, optionally against the same case in a baseline run. A case that also failed in the baseline is never blamed on the difference.
- FR-39: An **improvement loop** proposes a revision to one prompt for the leading cause, re-evaluates, and keeps it only if the primary metric improves significantly and no gated metric regresses. It changes **prompt text only**, as new immutable versions in a prompt store, and applies a revision only with approval. Every run records the prompt versions it used.
- FR-40: Sagwa's evaluation tools are available over **MCP**, each answering exactly as its CLI command does, so a coding agent can check its own change for regressions.

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Reproducibility** | Every run pins exact model snapshot IDs (not floating aliases like "latest"); temperature/seed are recorded per run |
| **Statistical honesty** | No aggregate metric is reported without an accompanying sample size and, where a comparison is made, a significance test — the system should never imply more precision than a small golden set supports |
| **Performance** | A 150-case golden set completes a run in under 5 minutes against a standard rate-limited API, via concurrent execution |
| **Cost transparency** | Every run reports total cost; the system supports routing judge calls to a cheaper model for high-volume, low-stakes tasks (e.g., clustering labels) without requiring separate code paths |
| **Auditability** | Every run is immutably tied to a git SHA and dataset version; historical runs are never mutated, only appended |
| **Extensibility** | Adding a new task type (metrics + target pipeline adapter) should not require changes to the storage schema or CLI |
| **Portability** | The system runs locally (Postgres via Docker Compose) with no required cloud dependency, so it's fully demoable offline. *Current interim default is SQLite with no service to run — see [ADR-0004](./adr/0004-sqlite-default-defer-postgres.md); Postgres remains the intended production-shaped store once clustering (Week 7-8) needs pgvector for embeddings.* |

---

## 7. Success Metrics

Since this is a portfolio/infrastructure project rather than a live product with users, success is measured against **demonstrable proof points**, not usage metrics.

**Every row here is tracked in [STATUS.md](./STATUS.md)** — a metric with no recorded evidence counts as unmet, not as done.

| Metric | Target |
|---|---|
| Judge–human agreement (Cohen's κ) on calibration set | ≥ 0.70 |
| Golden-set size used for calibration | ≥ 150 hand-labeled cases |
| Injected synthetic regressions correctly detected by the diff engine | 100% (this is a test of the tester — must not have false negatives) |
| False-positive rate on injected no-op changes (e.g., re-running the same pipeline twice) | Low enough to report an honest false-positive rate at the chosen significance threshold |
| CI gate demo | At least one real PR in the repo's history that was blocked by the gate and one that passed after a fix |
| Cost reduction from tiered model routing | **De-scoped ([ADR-0007](./adr/0007-descope-tiered-routing.md)).** Per-role routing shipped; cost-tiered routing waits for a judge that clears κ ≥ 0.70. Original target: documented % reduction in judge-call cost vs. a naive single-model baseline, at equivalent agreement |
| Judge quality vs. a naive uncalibrated judge | Sagwa's rubric-based judge and a one-line no-rubric judge scored against the same 200 human labels, with κ reported for both — the delta is the headline proof point for the whole project (FR-15a) |
| End-to-end demo | A recorded walkthrough: golden set → run → diff → clustered failures → CI gate → dashboard, in under 5 minutes, using the HotpotQA v1→v2 regression as the live example |

---

## 8. Scope: V1 vs Future Versions

### V1 (this build, 1–3 months)
- CLI-driven eval runner with two target task types: **multi-hop RAG-QA** (HotpotQA, shipped as a baseline and a regressed candidate) and **summarization** (CNN/DailyMail, for generalization proof)
- Adapter interface decoupling Sagwa from every target's internals — no shared code, no dependency on a target's release cadence
- RAGAS integration + custom LLM-judge harness with documented calibration, including a **head-to-head against a naive uncalibrated judge** on the same human labels
- Postgres-backed run history (pinning both Sagwa's and the target pipeline's git SHA per run) and diff engine with stat-sig testing
- HDBSCAN-based failure clustering with auto-labeling
- GitHub Actions CI gate
- Streamlit dashboard

### V2 / Future Directions (explicitly deferred, mention in writeup as forward-looking)

Nothing here starts before the v1 milestones in §10 are hit — this section exists so "good idea, later" doesn't quietly become scope creep (see §9's scope-creep risk).

| Initiative | What it would take | Why it's not v1 |
|---|---|---|
| **Online/production traffic sampling & shadow evaluation** | A live-traffic sampler plus an async eval path that doesn't block the request; schema changes to distinguish golden-set runs from sampled-production runs | v1 is explicitly offline/batch against golden sets (§3 non-goal) — a different execution model, not an extension of this one |
| **Multi-judge ensembling with disagreement-based active learning** | 2+ judges scored per case, a disagreement metric, and a queue surfacing high-disagreement cases for human labeling | Needs a working single-judge calibration first (FR-13..15) — a refinement on a result that doesn't exist yet |
| **Hosted, multi-tenant version with auth** | Auth, billing, tenant isolation — effectively a different product | Explicit §3 non-goal: a single-user/single-team tool, not a commercial platform |
| **Automatic golden-set expansion from production failure logs** | A pipeline mining flagged/low-score production cases, deduping against the existing golden set, routing candidates through human review | Depends on online sampling existing first |
| ~~**Agentic/tool-use evaluation**~~ **Built (2026-11), see §5.10 FR-29–FR-40.** Was: trajectory scoring, not just final-output scoring | A different metrics shape than any v1 task type — plus trace storage, trajectory diffing and root-cause clustering | Built after the v1 gaps closed; requirements in §5.10 |
| **Next.js dashboard rebuild** | Reimplementing the Streamlit v1 views | A stretch goal once the v1 dashboard exists — polish, not capability |
| **Langfuse tracing** | Wiring a tracer into the runner | Deferred mid-project, not future scope exactly — see [ADR-0002](./adr/0002-defer-langfuse.md). `results.trace_id` is already reserved for it |
| **Postgres + pgvector** | A `DATABASE_URL` change plus an embedding column | The intended production store, deferred only because SQLite needs no service — see [ADR-0004](./adr/0004-sqlite-default-defer-postgres.md). Revisit when clustering wants to persist embeddings |

**Extension points already built into v1** (no v2 work needed to use them): a new task type needs only metrics plus an adapter — `GoldenCase.task_type` and the `TargetAdapter` protocol are the seams, with no storage or CLI change; a new target pipeline needs no Sagwa-side change at all (`module.path:ClassName` loading); a new tracer needs no schema change (`results.trace_id` takes any tracer's ID as a string).

---

## 9. Risks & Open Questions

| Risk / Question | Notes |
|---|---|
| Will judge calibration reach κ ≥ 0.70? | If not achievable on the first task type, document the iteration process honestly rather than cherry-picking a passing metric — the iteration story is itself valuable |
| Is a 150-case golden set large enough for meaningful stat-sig testing? | Report confidence intervals transparently; this is a known limitation of small-N evals and should be stated, not hidden |
| Scope creep risk: building too many target task types | Hard cap at 2 task types for v1 (RAG-QA + summarization); document additional types as "extensibility, not delivered" |
| A target pipeline changes underneath Sagwa mid-project | Mitigated by FR-7a (pinning the target's git SHA per run) — a target-side change produces a new, distinguishable run rather than corrupting history. **Realized 2026-09-13**: the original target was removed entirely, and every run recorded against it stayed valid and attributable |
| Benchmark answers may be memorized by the judge or target model | HotpotQA is Wikipedia-derived, so a model can answer some questions without retrieval at all. This is why the regression candidate is measured on *faithfulness to retrieved context*, not only on answer correctness — a right answer that ignores the retrieved paragraphs is still a RAG regression |
| Public benchmarks are not a substitute for a golden set against your own traffic | Stated plainly in the writeup: these prove the machinery on real, messy inputs. A team adopting Sagwa still authors its own golden set — the loader, schema and adapter contract are what make that cheap |

---

## 10. Milestones

Live status for every milestone is tracked in [STATUS.md](./STATUS.md). At the PRD level, the three checkpoint deliverables are:

1. **End of Week 6**: Judge calibration report exists and is documented (κ score, confusion matrix, prompt iteration history) — this is the checkpoint that de-risks the rest of the project
2. **End of Week 10**: CI gate demonstrably blocks a real regressive PR in the repo
3. **End of Week 12**: Full demo walkthrough recorded, dashboard live, writeup published

---

## 11. Appendix: Glossary

- **Golden set**: a fixed, versioned dataset of inputs (and, where available, expected outputs/labels) used as the ground truth for evaluating a pipeline
- **Target pipeline**: the LLM-powered system under test (e.g., a RAG app, a summarizer)
- **LLM-as-judge**: using an LLM to score another LLM's output against a rubric, in place of (or alongside) human review
- **Calibration**: the process of validating that judge scores agree with human judgment, reported via agreement rate and Cohen's κ
- **Regression**: a statistically significant drop in a quality metric between two runs of the same golden set against different pipeline versions
- **Target pipeline**: any AI system Sagwa evaluates, reached only through the `TargetAdapter` contract. v1's targets are built from public benchmarks (`benchmarks/adapters.py`); Sagwa's own code knows none of them by name
