# Sagwa — LLM Evaluation & Regression-Testing Platform

**A CI/CD system for LLM quality.** Ship a prompt, model, or RAG-pipeline change and know — automatically, before it reaches users — whether it made things better or worse.

---

## 1. Problem Statement

Teams shipping LLM-powered features today mostly deploy on vibes: someone tweaks a prompt, skims five outputs, and merges. There is no equivalent of a unit-test suite for "did this change make the AI worse at its job." This is exactly the gap between a hobby project and a production AI system, and it's the single most common failure mode at companies scaling past their first AI feature — quality regresses silently, nobody notices until a customer complains, and there's no historical record to point to *when* or *why* it broke.

Sagwa closes that gap. It's not another AI *app* — it's the **testing and observability infrastructure that sits underneath** AI apps, the same way pytest + CI sits underneath a normal codebase. That's a deliberate positioning choice: most portfolio projects build a flashy demo; this builds the boring, load-bearing thing that senior/staff AI engineers are actually judged on — "can I trust what you ship."

### Concretely, Sagwa lets an engineer:
1. Define a **golden dataset** (inputs + expected properties/labels) for an LLM task (RAG QA, summarization, classification, agent tool-use, etc.)
2. Run that dataset against **any prompt/model/pipeline version** and get structured metrics back (faithfulness, relevance, toxicity, latency, cost, task-specific accuracy)
3. **Diff** two runs (e.g., prompt v13 vs v14) and see exactly which test cases regressed, with LLM-judge explanations
4. Gate a **CI pipeline** so a PR that drops quality below a threshold fails the build, just like a broken unit test
5. **Cluster failures by semantic similarity** to spot systemic issues (e.g., "23% of failures involve multi-hop questions") instead of reading transcripts one by one
6. Track **cost and latency per model/prompt version over time** on a dashboard

---

## 2. Why This Project (Positioning Rationale)

`★ Insight ─────────────────────────────────────`
Most AI portfolio projects answer "can you build an AI feature." Sagwa answers "can you be trusted to own the quality bar for AI features at a company" — a categorically different and more senior question. Product companies scaling AI features hit this exact wall (no regression testing → silent quality drift → customer-visible failures) within 6-12 months of shipping their first LLM feature, so this project maps directly onto a problem hiring managers have personally lived through, not one they're hypothetically worried about.
`─────────────────────────────────────────────────`

- **Uniqueness**: very few candidates build eval *tooling* — almost everyone builds an eval'd app. Tooling signals platform-engineering maturity.
- **Reusability**: this is infrastructure any team can point at any of their own LLM features — unlike a vertical demo (e.g., "invoice RAG"), it doesn't need translation to be relevant.
- **Interview leverage**: gives you a rich, specific story about calibration, statistics (agreement rates, confidence intervals), and CI/CD — topics that separate mid-level from senior conversations.

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         GOLDEN DATASETS                            │
│   YAML/JSONL: {input, expected_output|labels, task_type, tags}     │
│   versioned in git, per task (RAG-QA, summarization, classify...)  │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ↓
┌───────────────────────────────────────────────────────────────────┐
│                       EVAL RUNNER (batch)                          │
│  - loads target pipeline (prompt/model/RAG config under test)      │
│  - executes golden set concurrently (async batching, rate-limited) │
│  - captures: output, latency, token usage, cost, trace              │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ↓
┌───────────────────────────────────────────────────────────────────┐
│                         METRICS LAYER                              │
│  Reference-based:  exact/fuzzy match, ROUGE, embedding similarity   │
│  Reference-free:   RAGAS (faithfulness, context precision/recall)   │
│  LLM-as-judge:     task-specific rubric scoring (calibrated!)       │
│  Safety:           toxicity, PII leakage, refusal-rate checks       │
│  Ops:              p50/p95 latency, cost/request, token count       │
└───────────────────────────────┬─────────────────────────────────────┘
                                 ↓
┌───────────────────────────────────────────────────────────────────┐
│                    STORAGE + RUN HISTORY (Postgres)                │
│   runs(id, git_sha, prompt_version, model, dataset_version, ts)    │
│   results(run_id, case_id, output, metrics_json, trace_id)         │
└───────┬───────────────────────────────┬───────────────────────────┘
        ↓                               ↓
┌────────────────────┐      ┌─────────────────────────────────────┐
│   DIFF/REGRESSION    │      │   FAILURE CLUSTERING                │
│   engine — compares   │      │   embed failing cases → HDBSCAN/    │
│   run A vs run B,      │      │   k-means → auto-labeled clusters   │
│   flags stat-sig drops │      │   ("multi-hop questions", "long    │
│   per metric/tag       │      │    docs", "negation handling"...)  │
└────────┬────────────┘      └──────────────┬──────────────────────┘
         ↓                                  ↓
┌───────────────────────────────────────────────────────────────────┐
│                    CI GATE + DASHBOARD                             │
│  GitHub Action: `sagwa run --gate faithfulness>=0.85`         │
│  fails PR if gate violated → posts metric diff as PR comment        │
│  Dashboard (Streamlit/Next.js): trend lines, cost, cluster browser  │
└───────────────────────────────────────────────────────────────────┘
```

### Key design decisions and why

| Decision | Why |
|---|---|
| **Golden sets are versioned files in git, not a DB table** | Same review discipline as code — a change to expected behavior goes through a PR, is diffable, and is tied to a commit. This is the detail that makes it feel like real engineering, not a script. |
| **LLM-as-judge is *calibrated* against human labels, not trusted blindly** | The credibility of any eval system rests on knowing how much to trust it. You hand-label ~150-200 cases, compute judge-vs-human agreement (Cohen's κ), and report it — this single number is the most interview-relevant artifact in the whole project. |
| **Metrics stored per-case, not just aggregated** | Aggregates hide regressions in a subpopulation (e.g., avg faithfulness barely moves but multi-hop questions cratered). Per-case storage is what enables the clustering step and is a deliberate nod to how real eval infra (e.g., OpenAI evals, Braintrust) is built. |
| **Stat-sig testing on the diff, not just "score went down"** | With small golden sets, a 2-point metric drop can be noise. Using a paired bootstrap or McNemar's test for pass/fail metrics avoids shipping false regressions/false confidence — shows statistical maturity. |
| **CI gate as a CLI + GitHub Action, not just a dashboard** | A dashboard nobody checks doesn't prevent regressions. Gating the merge is what makes this *infrastructure* rather than a reporting tool — mirrors exactly how test coverage/lint gates work in normal SWE, translating your existing SWE instincts into the AI domain (a strong personal narrative for a SWE→AI pivot). |

---

## 4. Hugging Face / Model-Ecosystem Tasks Used

| Task | Where it's used |
|---|---|
| **Zero-Shot Classification** | Auto-tagging failure cases by type (e.g., "hallucination," "refusal," "format error") without training a classifier per project |
| **Sentence Similarity** | Embedding-based semantic diffing — clustering failures, and reference-free similarity scoring for outputs without exact-match ground truth |
| **Text Generation** | The LLM-as-judge itself, and the target pipelines being evaluated (RAG QA, summarization) used as example subjects for the eval suite |
| **Text Ranking / Text Classification** | Pairwise comparison mode ("is output A or B better") as an alternative judge protocol, more reliable than absolute scoring for subjective quality |

---

## 5. Tech Stack — and the reasoning behind each choice

| Layer | Choice | Why this over alternatives |
|---|---|---|
| **Eval framework** | **RAGAS** (for RAG-specific metrics) + a **custom judge harness** | RAGAS gives you faithfulness/context-precision/recall out of the box for RAG tasks — no need to reinvent well-studied metrics. But RAGAS alone doesn't cover CI gating, run history, or clustering, so a thin custom layer wraps it for orchestration. Building 100% from scratch would waste time reinventing solved metrics; using *only* an off-the-shelf tool (e.g., a SaaS eval product) would remove the systems-design work that makes this portfolio-worthy. This middle path is deliberate. |
| **LLM Judge model** | Claude or GPT-4o-class for judging, a cheaper model (Haiku/GPT-4o-mini) for high-volume classification/clustering tasks | Judge quality matters more than judge cost since it's the arbiter of truth — but tagging/clustering doesn't need frontier intelligence, so routing by task cost-optimizes the whole pipeline. This tiered-model routing is itself a talking point (shows cost-consciousness, not just "throw the biggest model at everything"). |
| **Orchestration/runner** | Python, `asyncio` + a task queue (or simple `httpx` + semaphore for concurrency control) | Golden sets need concurrent execution against rate-limited APIs. Async keeps this simple without pulling in a heavyweight distributed system — right-sized for a project run by one engineer, not over-engineered with Celery/Kafka for a workload that doesn't need it. |
| **Storage** | **Postgres** (via SQLAlchemy or just `psycopg`) | Run history is inherently relational (runs → results → cases, with joins for diffing). A vector-capable Postgres (pgvector) also lets you store failure-case embeddings in the same DB instead of standing up a separate vector store — one fewer moving part, which matters for a solo 1-3 month build. |
| **Clustering** | `sentence-transformers` embeddings + HDBSCAN | HDBSCAN doesn't require specifying cluster count upfront (unlike k-means) — appropriate since you don't know in advance how many failure modes exist. Auto-labels each cluster by asking the judge model to summarize its centroid-nearest cases. |
| **CI integration** | GitHub Actions + a small `sagwa` CLI (`pip install -e .`) | Makes the "gate" story concrete and demoable — a real PR, a real failing check, a real posted comment. This is more convincing in an interview than a slide describing the concept. |
| **Dashboard** | **Streamlit** for v1, optionally a Next.js rebuild later | Streamlit lets you get a working trend dashboard in days, not weeks — appropriate given the 1-3 month budget is better spent on the eval logic and calibration study (the parts that actually demonstrate judgment) than on frontend polish. A Next.js version is a good "if I have extra time" stretch goal, and rebuilding it also gives you a second, framework-agnostic talking point. |
| **Tracing** | Langfuse (self-hosted or free tier) | Rather than hand-rolling trace storage, Langfuse gives you span-level tracing, cost tracking, and a UI for free — reserving your engineering time for the eval/stats layer that's actually novel here, and demonstrating you know when to use existing LLMOps tooling vs. build custom (a real judgment call product companies care about). |
| **Stats** | `scipy.stats` (bootstrap resampling, McNemar's test) | Off-the-shelf, well-trusted implementations — no reason to hand-roll significance testing, and using a recognized method (vs. an ad hoc "3% is significant") is what makes the regression-gating claim credible under scrutiny. |

---

## 6. Data Plan

### Target Pipeline #1: **ringo** (existing project, real target — not built for this project)

Rather than standing up a throwaway RAG app just to have something to evaluate, **Sagwa's primary target pipeline is [ringo](../ringo)** — an already-built hybrid BM25+semantic RAG chat system (cross-encoder reranking, citation grounding, groundedness caveats). This is a deliberate, stronger choice than a synthetic target for one specific reason:

`★ Insight ─────────────────────────────────────`
ringo already ships an ad-hoc LLM-judge (`backend/eval.py`) — a single uncalibrated Groq call per metric, no run history, no significance testing. That's a **real prior baseline to beat**, not a hypothetical one. The calibration writeup can report an actual before/after: "ringo's existing judge vs. Sagwa's calibrated judge," which is a far more credible story than claiming an eval system is good in the abstract.
`─────────────────────────────────────────────────`

- **Integration boundary**: Sagwa and ringo stay architecturally separate — Sagwa treats ringo purely as an external **target pipeline under test**, calling its FastAPI endpoint (or importing `pipeline.py` directly) through a thin adapter that returns `{answer, context, latency_ms, tokens, cost}` in Sagwa's expected shape. Sagwa's code makes zero assumptions about ringo internals beyond that adapter contract — this mirrors how eval infra works against a service at a real company (you rarely own the thing you're evaluating) and keeps both projects independently portable/demoable.
- **Reproducibility**: pin the exact ringo git SHA being evaluated in every Sagwa run record — ringo is an actively-developed repo, so a floating target would break the "immutable run history" guarantee (see PRD NFR "Reproducibility").
- **Golden set**: still needs to be hand-built — ringo doesn't ship QA pairs against its own document corpus. Load a real, varied document set into ringo's `documents/` (a mix of PDF/DOCX/PPTX/CSV to exercise its multi-format ingestion), then hand-write ~100-150 `(question, expected_answer_or_key_facts, tags)` pairs against that corpus. This is unavoidable work regardless of target pipeline choice — budget the full 2 weeks in §9.
- **Human labels for judge calibration**: hand-label ~150-200 `(query, context, answer, correct?)` triples from real ringo runs — the calibration section is the credibility anchor of the whole project; don't skip or rush it.
- **Synthetic adversarial cases**: construct ~30-50 edge cases (negation, ambiguous references, out-of-scope questions, queries targeting the groundedness-caveat path) to stress-test both ringo's retrieval and Sagwa's judge — label clearly as synthetic in the writeup so it isn't conflated with the real-corpus claim.

### Target Pipeline #2: a public-dataset task (for breadth)

Keep one additional target task type — e.g., summarization over CNN/DailyMail or arXiv abstracts — evaluated with a minimal reference implementation. This exists purely to prove Sagwa generalizes beyond one pipeline (FR-requirement: "adding a new task type shouldn't require storage/CLI changes"), not to be a polished app in its own right.

---

## 7. Evaluation Methodology (the part that must be rock solid)

1. **Judge calibration study**: hand-label a stratified sample, compute judge-vs-human agreement (accuracy + Cohen's κ), report a confusion matrix, and **iterate the judge prompt until κ ≥ 0.7** before trusting it for anything downstream. Document every iteration — this is genuinely the most valuable artifact in the repo.
2. **Regression detection validation**: inject known synthetic regressions (e.g., truncate context, swap in a worse model) and confirm the system actually flags them — a "test for your test."
3. **Statistical rigor**: report confidence intervals on aggregate metrics (bootstrap over the golden set), not bare point estimates — small golden sets (50-150 cases) have real sampling noise that a serious eval system must acknowledge.
4. **Metric selection per task type**: don't force one metric onto everything — RAG gets faithfulness/context precision, classification gets accuracy/F1, generation gets judge-rubric scores. Document the mapping explicitly.

---

## 8. LLMOps / Production Concerns

- **Cost tracking**: every run logs token counts × current pricing → cost dashboard, with tiered-model routing shown to cut judge cost materially (this becomes a resume metric).
- **Latency budget**: batch eval runs report p50/p95 so a slow pipeline change is caught alongside a low-quality one.
- **Guardrail checks**: a lightweight PII/toxicity scan on golden-set outputs, since eval systems themselves can leak sensitive info if the target pipeline does.
- **Reproducibility**: pin model versions/temperature=0 (or report variance across seeds) per run so a "regression" isn't just sampling noise — pin exact model snapshot IDs, not just "gpt-4o," since silent provider-side model updates are a real, underappreciated cause of production drift.

---

## 9. Timeline (1–3 months, solo)

| Weeks | Focus |
|---|---|
| 1–2 | Golden dataset schema + loader, pick target pipelines (RAG-QA + summarization), stand up Postgres schema |
| 3–4 | Eval runner (async batch execution), RAGAS integration, reference-based metrics |
| 5–6 | LLM-as-judge + **calibration study** (this is the critical-path deliverable — budget real time here) |
| 7–8 | Diff engine + stat-sig testing, failure clustering (HDBSCAN + auto-labeling) |
| 9–10 | CI gate (GitHub Action + CLI), Langfuse tracing integration, cost/latency tracking |
| 11 | Streamlit dashboard (trend lines, cluster browser, run diff view) |
| 12 | Write-up (blog post walking through the calibration study specifically), demo video, polish README |

---

## 10. Repo Structure (proposed)

```
Sagwa/
├── README.md                   # public-facing project summary (write after MVP)
├── pyproject.toml
├── sagwa/
│   ├── cli.py                  # `sagwa run`, `sagwa diff`, `sagwa gate`
│   ├── datasets/                # golden set loaders + schema (pydantic models)
│   ├── runner/                  # async batch execution against target pipelines
│   ├── metrics/                 # ragas wrappers, judge harness, safety checks
│   ├── judge/                   # judge prompts + calibration scripts
│   ├── diff/                    # regression detection, stat-sig tests
│   ├── clustering/               # embedding + HDBSCAN + auto-labeling
│   ├── storage/                  # SQLAlchemy models, migrations
│   └── dashboard/                 # Streamlit app
├── golden_sets/
│   ├── rag_qa_hotpotqa.jsonl
│   ├── summarization_cnn_dm.jsonl
│   └── adversarial_edge_cases.jsonl
├── calibration/
│   ├── human_labels.csv
│   └── calibration_report.md    # κ scores, confusion matrix, judge prompt iterations
├── targets/                      # example "apps under test" (RAG pipeline, summarizer)
├── .github/workflows/eval-gate.yml
└── docs/
    ├── PRD.md
    ├── PLAN.md                   # this file
    ├── writeup.md                # the blog-post-style deep dive for your portfolio
    └── adr/                      # architecture decision records
```

---

## 11. Resume-Ready Impact Metrics (fill in with real numbers post-build)

- *"Built an LLM eval CI pipeline gating N prompt/model deploys across 2 pipelines, catching X quality regressions pre-release."*
- *"Calibrated an LLM-as-judge against Y hand-labeled examples, achieving Z% agreement (Cohen's κ = 0.NN) with human raters."*
- *"Cut judge-model inference cost by N% via tiered model routing without degrading agreement scores."*
- *"Automated failure clustering across M failed test cases into K semantically distinct failure modes, cutting manual triage time from hours to minutes."*
- *"Reduced mean regression-detection time from ad hoc manual review to <N minutes via CI-gated automated evals."*

---

## 12. Risks / What Could Go Wrong (and mitigations)

| Risk | Mitigation |
|---|---|
| Judge calibration never reaches acceptable agreement | Budget 2 full weeks, not days; iterate rubric wording, few-shot examples in judge prompt, and consider ensembling 2 judges |
| Golden sets too small for statistically meaningful diffs | Report confidence intervals honestly rather than overclaiming; note this limitation explicitly in the writeup — reviewers respect honesty about statistical power more than false precision |
| Scope creep into building 5 target pipelines instead of 2 | Timebox to exactly 2 target pipeline types (RAG-QA, summarization) — the eval infra is the product, not the pipelines under test |
| Project reads as "just another RAGAS wrapper" | The calibration study, stat-sig diffing, and CI gate are the differentiators — lead with those in the writeup, not with "I used RAGAS" |

---

## 13. Next Steps

1. Scaffold the repo structure above
2. Pick and download the two target-pipeline datasets (HotpotQA subset, CNN/DailyMail subset)
3. Define the golden-set schema (pydantic model) — this is the foundational contract everything else builds on
4. Build the async eval runner against a trivial pipeline first (even a stub) to validate the storage schema before investing in metrics
