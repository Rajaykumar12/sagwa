"""Judge-based metric (PRD FR-10, and a prerequisite for the dashboard's
FR-28 case drill-down): scores each case's answer via the LLM-as-judge
harness (`sagwa/judge/harness.py`) and stores both the parsed score and
the judge's raw rationale text under `metrics_json["judge"]`.

Requires `GROQ_API_KEY`. Mirrors the "degrade gracefully rather than crash
the run" pattern used by `sagwa/metrics/ragas_metrics.py` and
`embedding_similarity()`: when no key is configured, `compute_judge_metric`
returns `None` and the case simply has no `judge` key, rather than failing
the whole run.
"""
from __future__ import annotations

import os

_llm_call = None


def compute_judge_metric(
    query: str,
    answer: str,
    expected: str | None = None,
    context: str | None = None,
) -> dict | None:
    """Returns `{"score": float|None, "rationale": str, "prompt_version": str}`,
    or `None` when `GROQ_API_KEY` isn't configured — not attempted, not a
    silent 0.0.

    `expected` and `context` are passed through to the judge so it can grade
    correctness rather than fluency: without them the judge scored a known-worse
    pipeline 0.22 higher than its baseline (docs/STATUS.md §2a).

    `prompt_version` is stored per case because a judge score is only
    comparable to others from the same prompt — and only trustworthy up to the
    kappa measured for that version (FR-14, FR-15)."""
    if "GROQ_API_KEY" not in os.environ:
        return None

    from sagwa.judge.harness import JUDGE_PROMPT_VERSION, groq_llm_call, score_absolute_with_rationale

    global _llm_call
    try:
        if _llm_call is None:
            _llm_call = groq_llm_call()
        score, rationale = score_absolute_with_rationale(
            _llm_call, query=query, answer=answer, expected=expected, context=context
        )
        return {"score": score, "rationale": rationale, "prompt_version": JUDGE_PROMPT_VERSION}
    except Exception as e:
        return {"score": None, "rationale": None, "_error": str(e)}
