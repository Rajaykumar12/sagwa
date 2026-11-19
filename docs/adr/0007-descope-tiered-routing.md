# ADR 0007: per-role model routing ships; cost-tiered routing is de-scoped

**Status**: Accepted (2026-11-19)

## Context

PRD §7 lists a success metric: *"Cost reduction from tiered model routing: documented % reduction in judge-call cost vs. a naive single-model baseline, at equivalent agreement."* No implementation of it ever existed, and until 2026-09-20 it appeared in no status table. It was invisible, not deprioritized.

Two different things go by "routing":

1. **Per-role routing**: each LLM role (judge, RAGAS, cluster labelling) uses its own configurable model: `SAGWA_JUDGE_MODEL`, `SAGWA_RAGAS_MODEL`, `SAGWA_CLUSTER_MODEL`. This shipped on 2026-09-23. Cheap, low-stakes calls (cluster labels) go to a small model, and each role draws on its own per-model rate limit, which on a free tier matters more than price (docs/STATUS.md §3a).
2. **Cost-tiered routing**: a cheap judge scores every case and only uncertain cases escalate to an expensive one. This is what the §7 metric measures.

## Decision

**Keep per-role routing (1). De-scope cost-tiered routing (2) and its §7 metric.**

Reasons:

- **No judge it could route between clears the bar.** On 100 held-out human labels the best measured judge reaches κ = 0.66 (a naive prompt on `gpt-oss-120b`); the rubric judges reach 0.32–0.46 (`calibration/calibration_report.md` §7). The metric requires "equivalent agreement", and equivalence to a judge that fails its own bar is not a meaningful target. Making an unreliable judge cheaper solves the wrong problem.
- **The judge isn't where the cost is.** A 50-case RAG run spends most of its ~170K tokens on RAGAS faithfulness, not on judge calls (STATUS.md §3a). The lever that cut cost was `SAGWA_RAGAS_METRICS`, which turns RAGAS from two metrics into one or none. That shipped.
- **Escalating on judge confidence needs a calibrated confidence signal.** The judge emits a score on a coarse band scale, not a probability, so "escalate when unsure" would itself need calibrating against the same labels, which is a second study on top of the one that failed.

## Consequences

- PRD §7's routing row is marked **de-scoped (ADR-0007)**, not left red with no owner.
- Revisit once a judge clears κ ≥ 0.70. At that point "same agreement for less money" becomes a real question, and the per-role env vars are already the seam a router would plug into.
- The agentic expansion (docs/AGENTIC_PRD.md) measures cost per successful task as a *target* metric. That is about the system under test, not Sagwa's own judge spend, and is unaffected.
