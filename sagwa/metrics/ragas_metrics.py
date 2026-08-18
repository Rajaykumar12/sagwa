"""RAGAS reference-free RAG metrics (PRD FR-9): faithfulness and context
precision. Only meaningful for `task_type == "rag_qa"` cases that have
`context` (the adapter-reported retrieved context).

Requires the `metrics` extra (`ragas`, `langchain-groq`) and a configured
GROQ_API_KEY — reusing the same provider ringo's own `backend/eval.py`
uses, which is what keeps the calibration head-to-head comparison
apples-to-apples (PRD FR-15a, PLAN.md §5).

Known current blocker (see docs/GAP_ANALYSIS.md): `ragas==0.4.3` fails to
import in this project's `.venv` — it reaches for
`langchain_community.chat_models.vertexai`, which doesn't exist in the
installed `langchain-community==0.4.2`. That's a dependency-pinning
mismatch to resolve (upgrade/downgrade one side), not something this
module can work around, so every metric here degrades to `None` with an
`_error` string rather than crashing the case's other metrics.
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

        # Model choice mirrors sagwa/judge/harness.py::groq_llm_call — see
        # its docstring for why "llama-3.1-8b-instant" (ringo's eval.py
        # default) is no longer viable. reasoning_effort="low" matters for
        # the same reason it does there: gpt-oss models spend max_tokens on
        # hidden reasoning first, so a low-effort setting is needed to get
        # a usable score back within any reasonable token budget. Untested
        # end-to-end here — `ragas` itself fails to import, see module
        # docstring — but kept consistent with the path that is verified.
        chat = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.environ["GROQ_API_KEY"],
            model_kwargs={"reasoning_effort": "low"},
        )
        _judge_llm = LangchainLLMWrapper(chat)
    return _judge_llm


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
        try:
            results[key] = _score(metric_cls(llm=llm), sample)
        except Exception as e:
            results[key] = None
            results[f"{key}_error"] = str(e)
    return results
