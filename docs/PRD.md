# Product Requirements Document: Sagwa

**LLM Evaluation & Regression-Testing Platform**

| | |
|---|---|
| **Author** | Rajay Kumar |
| **Status** | Draft — v1.0 |
| **Last updated** | 2026-08-15 |
| **Related doc** | [PLAN.md](./PLAN.md) — technical architecture & tech-stack rationale |

---

## 1. Summary

Sagwa is CI/CD infrastructure for LLM quality. It lets an engineer define a golden test set for an LLM-powered task, run it against any prompt/model/pipeline version, and get back structured, statistically defensible metrics — with the ability to gate a merge in CI the same way a failing unit test would. It exists to replace "vibes-based" LLM deployment with a testable, auditable process.

This PRD defines *what* is being built and *why it matters to a user*, in product terms. See PLAN.md for the technical architecture, tech-stack rationale, and week-by-week build plan.

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
- Building or maintaining target pipelines to production quality as part of *this* project — the primary target, [ringo](../ringo), already exists as a separately-maintained, real project; Sagwa depends on it only through a stable adapter contract and does not take on responsibility for its quality or roadmap

---

## 4. Users & Use Cases

### Primary persona: **Dana, AI Engineer**
Owns a RAG-based support-doc assistant at a mid-size SaaS company. Wants to try a new retrieval strategy but is worried it'll quietly make answers less faithful to source docs. Currently has no way to check this except manually reading 20 outputs and guessing.

**Use case**: Dana runs `sagwa run --target rag_v2 --dataset golden/support_qa.jsonl`, gets faithfulness/context-precision scores, then runs `sagwa diff --baseline rag_v1 --candidate rag_v2` and sees a table of exactly which questions got worse, each with a judge-generated explanation.

**Concrete v1 instantiation**: this persona is realized directly, not simulated — Sagwa's actual primary target pipeline is **ringo** (an existing hybrid BM25+semantic RAG chat app in this workspace), evaluated through a thin external adapter with zero coupling to ringo's internals. ringo already contains an uncalibrated, ad-hoc LLM-judge (`backend/eval.py`); Sagwa's calibration study produces a direct, honest before/after comparison against it — a real regression story, not a hypothetical one. See [PLAN.md §6](./PLAN.md#6-data-plan) for the integration boundary and reproducibility requirements (pinned ringo git SHA per run).

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
- FR-3b: The reference v1 adapter integrates **ringo** (external RAG chat app, see [ringo](../ringo)) by calling its FastAPI endpoint; Sagwa's code makes no assumptions about ringo internals beyond this contract
- FR-3c: A second adapter (public-dataset summarization task) exists solely to prove the adapter interface generalizes across task types without storage/CLI changes

### 5.3 Eval Execution
- FR-4: A user can run a named golden set against a named target pipeline via CLI: `sagwa run --target <name> --dataset <path>`
- FR-5: Execution is concurrent/batched with configurable rate limiting to respect API limits
- FR-6: Each run captures, per case: raw output, latency, token usage, estimated cost, and a trace ID
- FR-7: A run is uniquely identified and persisted with its git SHA, prompt version, model identifier, and dataset version
- FR-7a: Where the target pipeline is an external, independently-versioned system (e.g., ringo), the run record additionally pins the **target pipeline's own git SHA** — not just Sagwa's — since the two repos version independently and a regression could originate on either side

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
- FR-15a: The calibration report supports an explicit **baseline comparison mode** — scoring a prior/external judge (e.g., ringo's existing `eval.py`) against the same human-labeled set, so Sagwa's judge can be reported against a real prior baseline, not just an absolute κ number

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
| **Portability** | The system runs locally (Postgres via Docker Compose) with no required cloud dependency, so it's fully demoable offline |

---

## 7. Success Metrics

Since this is a portfolio/infrastructure project rather than a live product with users, success is measured against **demonstrable proof points**, not usage metrics:

| Metric | Target |
|---|---|
| Judge–human agreement (Cohen's κ) on calibration set | ≥ 0.70 |
| Golden-set size used for calibration | ≥ 150 hand-labeled cases |
| Injected synthetic regressions correctly detected by the diff engine | 100% (this is a test of the tester — must not have false negatives) |
| False-positive rate on injected no-op changes (e.g., re-running the same pipeline twice) | Low enough to report an honest false-positive rate at the chosen significance threshold |
| CI gate demo | At least one real PR in the repo's history that was blocked by the gate and one that passed after a fix |
| Cost reduction from tiered model routing | Documented % reduction in judge-call cost vs. a naive single-model baseline, at equivalent agreement |
| Judge quality vs. ringo's existing ad-hoc judge (`backend/eval.py`) | Sagwa's calibrated judge scored against the same human-labeled set as ringo's uncalibrated single-call judge, with agreement (κ) reported for both — the delta is the headline proof point for the whole project |
| End-to-end demo | A recorded walkthrough: golden set → run → diff → clustered failures → CI gate → dashboard, in under 5 minutes, using ringo as the live target pipeline |

---

## 8. Scope: V1 vs Future Versions

### V1 (this build, 1–3 months)
- CLI-driven eval runner with two target pipelines: **ringo** (real, external, hybrid RAG chat app — primary target) and a minimal public-dataset summarization task (secondary, for generalization proof)
- Adapter interface decoupling Sagwa from ringo's internals — no shared code, no dependency on ringo's release cadence
- RAGAS integration + custom LLM-judge harness with documented calibration, including a **head-to-head comparison against ringo's existing ad-hoc judge**
- Postgres-backed run history (pinning both Sagwa's and the target pipeline's git SHA per run) and diff engine with stat-sig testing
- HDBSCAN-based failure clustering with auto-labeling
- GitHub Actions CI gate
- Streamlit dashboard

### V2 / Future Directions (explicitly deferred, mention in writeup as forward-looking)
- Online/production traffic sampling and shadow evaluation (not just offline golden sets)
- Multi-judge ensembling with disagreement-based active learning to prioritize what to hand-label next
- A hosted, multi-tenant version with auth (would turn this from a tool into a product — separate scope)
- Automatic golden-set expansion by mining real production failure logs
- Support for agentic/tool-use task evaluation (trajectory scoring, not just final-output scoring)

---

## 9. Risks & Open Questions

| Risk / Question | Notes |
|---|---|
| Will judge calibration reach κ ≥ 0.70? | If not achievable on the first task type, document the iteration process honestly rather than cherry-picking a passing metric — the iteration story is itself valuable |
| Is a 150-case golden set large enough for meaningful stat-sig testing? | Report confidence intervals transparently; this is a known limitation of small-N evals and should be stated, not hidden |
| Scope creep risk: building too many target task types | Hard cap at 2 target pipelines for v1 (ringo + one public-dataset task); document additional types as "extensibility, not delivered" |
| ringo changes underneath Sagwa mid-project (it's an actively-developed, separate repo) | Mitigated by FR-7a (pinning ringo's git SHA per run) — a ringo-side change simply produces a new, distinguishable run rather than corrupting history; if ringo's adapter-facing interface (its API shape) changes, the adapter needs updating, which is scoped as expected integration maintenance, not a project risk |
| Golden-set questions may accidentally leak into how ringo's documents were chosen, biasing results favorably | Write the golden-set questions *after* loading a fixed, unfamiliar document set into ringo — don't cherry-pick documents to match easy questions |

---

## 10. Milestones

See [PLAN.md §9](./PLAN.md#9-timeline-13-months-solo) for the detailed week-by-week technical build plan. At the PRD level, the three checkpoint deliverables are:

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
- **ringo**: an existing, separately-maintained hybrid BM25+semantic RAG chat application ([../ringo](../ringo)) used as Sagwa's primary real-world target pipeline, integrated purely through an external adapter
