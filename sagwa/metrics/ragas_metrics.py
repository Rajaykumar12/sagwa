"""RAGAS reference-free RAG metrics (PRD FR-9): faithfulness and context
precision. Only meaningful for `task_type == "rag_qa"` cases that have
`context` (the adapter-reported retrieved context).

Requires the `metrics` extra (`ragas`, `langchain-groq`) and a configured
GROQ_API_KEY — the same provider the judge harness uses, so RAGAS and the
judge stay comparable on cost and latency.

Previously blocked (see docs/STATUS.md), now fixed: `ragas==0.4.3`
unconditionally imports `ChatVertexAI` from
`langchain_community.chat_models.vertexai` at module load, a module
`langchain-community` removed as of `0.4.x` while sunsetting legacy
provider integrations — `ragas` declares no version bound on
`langchain-community`, so an unpinned install grabbed the latest and
broke. Fixed by pinning `langchain-community<0.4` in `pyproject.toml`'s
`metrics` extra (resolves to `0.3.31`, which still has the module and is
compatible with everything else installed). Live-verified end-to-end
against Groq (2026-08-19): real faithfulness/context-precision scores
returned, not just an import that succeeds.

Two optional env vars, both defaulting to the behavior above:
`SAGWA_RAGAS_MODEL` (the Groq model RAGAS scores with) and
`SAGWA_RAGAS_METRICS` (comma-separated subset to compute). Both exist for
Groq's per-model rate limits: scoring on a different model than the judge
uses gives RAGAS its own tokens-per-minute budget.

Every metric here still degrades to `None` with an `_error` string on any
failure (missing `GROQ_API_KEY`, a future API change, etc.) rather than
crashing the case's other metrics — that fallback behavior is unrelated
to the import bug above and stays regardless.
"""
from __future__ import annotations

import asyncio
import os

_judge_llm = None


def _get_judge_llm():
    """Lazily build the LLM RAGAS uses to score faithfulness/precision."""
    global _judge_llm
    if _judge_llm is None:
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        # Model choice mirrors sagwa/judge/harness.py::groq_llm_call.
        # reasoning_effort="low" matters for
        # the same reason it does there: gpt-oss models spend max_tokens on
        # hidden reasoning first, so a low-effort setting is needed to get
        # a usable score back within any reasonable token budget. Passed as
        # its own field, not inside model_kwargs — the installed
        # langchain-groq (1.1.3) promoted reasoning_effort to a first-class
        # ChatGroq field and rejects it inside model_kwargs with a
        # pydantic validation error. Verified live against Groq.
        chat = ChatGroq(
            model=os.environ.get("SAGWA_RAGAS_MODEL", "openai/gpt-oss-20b"),
            api_key=os.environ["GROQ_API_KEY"],
            reasoning_effort="low",
            # See sagwa/judge/harness.py::groq_llm_call — a run with no
            # request timeout hangs indefinitely when the connection dies.
            timeout=90.0,
        )
        _judge_llm = LangchainLLMWrapper(chat)
    return _judge_llm


def _enabled_metrics() -> set[str]:
    """Which RAGAS metrics to compute. `SAGWA_RAGAS_METRICS` (comma-separated)
    narrows the default set — a run whose gate config doesn't read
    `context_precision` can skip it and its LLM calls with it, which matters
    against a per-model tokens-per-minute rate limit."""
    configured = os.environ.get("SAGWA_RAGAS_METRICS")
    if not configured:
        return {"faithfulness", "context_precision"}
    return {name.strip() for name in configured.split(",") if name.strip()}


def _score(metric, sample):
    # ragas has used both a sync `single_turn_score` and an async
    # `single_turn_ascore` across versions; support either rather than
    # guessing wrong against a pinned version.
    if hasattr(metric, "single_turn_score"):
        return metric.single_turn_score(sample)
    return asyncio.run(metric.single_turn_ascore(sample))


def compute_ragas_metrics(query: str, context: str, answer: str) -> dict:
    """Returns {"faithfulness": float|None, "context_precision": float|None},
    plus an `_error` key per metric that failed."""
    if not context:
        return {"faithfulness": None, "context_precision": None}

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
    except ImportError as e:
        return {"faithfulness": None, "context_precision": None, "_error": f"ragas unavailable: {e}"}

    try:
        llm = _get_judge_llm()
    except Exception as e:
        return {"faithfulness": None, "context_precision": None, "_error": f"judge LLM unavailable: {e}"}

    sample = SingleTurnSample(user_input=query, retrieved_contexts=[context], response=answer)

    results: dict = {}
    for key, metric_cls in [
        ("faithfulness", Faithfulness),
        ("context_precision", LLMContextPrecisionWithoutReference),
    ]:
        if key not in _enabled_metrics():
            continue
        try:
            results[key] = _score(metric_cls(llm=llm), sample)
        except Exception as e:
            results[key] = None
            results[f"{key}_error"] = str(e)
    return results
