"""LLM-as-judge harness (PRD FR-10, FR-11): rubric scoring in absolute mode
(score one output 0.0-1.0) and pairwise mode (compare two outputs, A/B/tie).

The LLM call is injected as `LLMCall` (prompt -> raw completion text) so
`score_absolute`/`score_pairwise` are unit-testable without a network call
or GROQ_API_KEY; production code passes `groq_llm_call()`.
"""
from __future__ import annotations

import os
import re
from typing import Callable

LLMCall = Callable[[str], str]  # prompt -> raw text completion


def groq_llm_call(model: str = "openai/gpt-oss-20b") -> LLMCall:
    """Builds a real Groq-backed `LLMCall`. Requires GROQ_API_KEY — reuses
    the same provider ringo's own `backend/eval.py` uses, so the calibration
    head-to-head (PRD FR-15a) compares judges on equal footing.

    Default model verified live against Groq's `/models` endpoint
    (2026-08-18) — `llama-3.1-8b-instant`, the model ringo's own
    `backend/eval.py` hardcodes, has since been deprecated and 404s. That's
    a pre-existing issue in ringo, out of scope to fix here (FR-3a: zero
    changes to the target's codebase), but it means the FR-15a baseline
    comparison against ringo's judge will fail until ringo's own model
    string is updated."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
            # gpt-oss models spend max_tokens on hidden reasoning before
            # `content` — without this, a short max_tokens budget (needed
            # for a one-line score) leaves `content` empty. Verified live.
            reasoning_effort="low",
        )
        return response.choices[0].message.content.strip()

    return _call


DEFAULT_RUBRIC = (
    "Score how well the answer addresses the question, on a scale of "
    "0.0 (does not address it at all) to 1.0 (fully and correctly addresses it)."
)

_ABSOLUTE_PROMPT = (
    "You are grading an AI assistant's answer.\n\n"
    "Question: {query}\n"
    "Answer: {answer}\n\n"
    "{rubric}\n\n"
    "Respond with only a number between 0.0 and 1.0."
)

_PAIRWISE_PROMPT = (
    "You are comparing two AI assistant answers to the same question.\n\n"
    "Question: {query}\n"
    "Answer A: {answer_a}\n"
    "Answer B: {answer_b}\n\n"
    "{rubric}\n\n"
    "Respond with only 'A', 'B', or 'tie'."
)


def score_absolute(
    llm_call: LLMCall,
    query: str,
    answer: str,
    rubric: str = DEFAULT_RUBRIC,
) -> float | None:
    """Absolute rubric score in [0, 1], or `None` if the judge's response
    couldn't be parsed as a number in range — never silently coerced to 0.0,
    which would misreport "unparseable" as "worst possible score"."""
    prompt = _ABSOLUTE_PROMPT.format(query=query, answer=answer, rubric=rubric)
    return _parse_score(llm_call(prompt))


def score_pairwise(
    llm_call: LLMCall,
    query: str,
    answer_a: str,
    answer_b: str,
    rubric: str = DEFAULT_RUBRIC,
) -> str | None:
    """Returns `"A"`, `"B"`, `"tie"`, or `None` if unparseable."""
    prompt = _PAIRWISE_PROMPT.format(query=query, answer_a=answer_a, answer_b=answer_b, rubric=rubric)
    raw = llm_call(prompt).strip().lower()
    if raw.startswith("a"):
        return "A"
    if raw.startswith("b"):
        return "B"
    if "tie" in raw:
        return "tie"
    return None


def _parse_score(raw: str) -> float | None:
    match = re.search(r"\b(1(?:\.0+)?|0(?:\.\d+)?)\b", raw)
    return float(match.group()) if match else None
