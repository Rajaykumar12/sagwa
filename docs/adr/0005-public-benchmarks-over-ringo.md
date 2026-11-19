# ADR 0005: public benchmarks replace ringo as the reference target pipeline

**Status**: Accepted (2026-09-20). Supersedes [ADR-0003](./0003-ringo-adapter-in-process.md).

## Context

PRD v1 named **ringo** — a separately-maintained hybrid RAG chat app — as Sagwa's primary target pipeline, for one good reason: ringo shipped its own uncalibrated ad-hoc judge (`backend/eval.py`), which made it a *real prior baseline to beat* rather than a hypothetical one. FR-15a, PRD §7's headline success metric, and the recorded demo were all specified against it.

Three things broke that plan:

1. **The adapter was removed** (commit `58607f2`, 2026-09-13) — Sagwa's target-loading mechanism (`module.path:ClassName`) made a bundled, repo-specific adapter unnecessary, and keeping one implied a coupling the architecture explicitly rejects.
2. **ringo's judge model no longer exists.** `backend/eval.py` hardcodes `llama-3.1-8b-instant`, which Groq has deprecated; it 404s live. The head-to-head could not run without editing ringo's own source, which FR-3a forbids.
3. **The dependency made the requirements fragile.** Four documents (PRD, PLAN, writeup, ADR-0003) silently became fiction when one commit dropped one adapter, because success had been defined in terms of a *named system* rather than a *capability*.

Meanwhile the real gap that ringo was meant to close — "no real target has ever run end to end" — stayed open regardless, because ringo itself had never been run from Sagwa.

## Decision

**Reference target pipelines are built from public benchmarks, in-repo, under `benchmarks/`:**

- `HotpotV1` / `HotpotV2` — multi-hop RAG-QA over a pooled 492-paragraph HotpotQA corpus. v1 is the baseline (dense retrieval, top_k=3, grounded prompt); v2 is a deliberate "cost-cutting PR" (top_k=1, no refusal instruction). The pair exists so the diff and gate have a **genuine regression to catch**, not a synthetic one.
- `CnnDmSummarizer` — summarization against human-written reference summaries (FR-3c's generalization proof).
- `benchmarks/build.py` regenerates every golden set and label file from Hugging Face with a fixed seed, so the data is reproducible without vendoring it.

**FR-15a's baseline becomes a naive judge prompt, not an external system.** Sagwa's rubric-based judge is scored against a one-line, no-rubric judge on the same 200 human labels (HelpSteer2). This tests the same underlying claim — *calibration, not model choice, is what makes a judge trustworthy* — while depending on nothing outside this repo.

## Consequences

- **Requirements now describe capabilities, not systems.** A target pipeline can be deleted without invalidating the spec.
- **Reproducible by anyone.** A reader with a Groq key can rerun the whole proof; nobody could rerun the ringo comparison, since ringo isn't public.
- **Weaker in one specific way, and it should be said plainly**: ringo's `eval.py` was a real judge someone actually shipped. A naive prompt is a *stand-in* for that class of judge, chosen to be representative rather than strawmanned — it uses the same model and temperature, differing only in rubric and reasoning budget. The claim is about calibration, and that is the claim the writeup should make.
- **Public benchmarks are not a golden set against production traffic.** HotpotQA is Wikipedia-derived, so a model can sometimes answer without retrieval at all; this is why the regression candidate is judged on faithfulness to retrieved context, not answer correctness alone. Any team adopting Sagwa still authors its own golden set.
- ADR-0003 is superseded but kept: its finding (ringo's HTTP handler strips `context`) is why `AdapterResult` carries `context` for every adapter.
