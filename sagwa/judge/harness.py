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


DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"


def groq_llm_call(model: str | None = None) -> LLMCall:
    """Builds a real Groq-backed `LLMCall`. Requires GROQ_API_KEY.

    The model is `SAGWA_JUDGE_MODEL`, else `DEFAULT_JUDGE_MODEL`. Judging,
    RAGAS scoring (`SAGWA_RAGAS_MODEL`) and cluster labelling
    (`SAGWA_CLUSTER_MODEL`) are three different jobs: the judge arbitrates
    quality and deserves the better model, while naming a cluster of failures
    does not. Routing them separately is what makes the per-role cost
    comparison in PRD §7 measurable — and, on a rate-limited tier, gives each
    role its own per-model budget.

    The FR-15a baseline judge runs through this same function with the same
    model and temperature, differing only in prompt — so the comparison
    isolates calibration, not model choice (see
    docs/adr/0005-public-benchmarks-over-ringo.md)."""
    from groq import Groq

    # timeout: without one, a dropped connection (laptop suspend, wifi flap)
    # leaves a run blocked forever on a socket that will never answer —
    # observed twice during the 2026-09-20 benchmark runs.
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=90.0)
    resolved_model = model or os.environ.get("SAGWA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            # Even at reasoning_effort="low", hidden reasoning can exceed a
            # small budget: at 60, 4/10 HelpSteer2 calibration cases came
            # back with empty `content` (a None score). Verified live.
            max_tokens=512,
            temperature=0.0,
            # gpt-oss models spend max_tokens on hidden reasoning before
            # `content` — without this, a short max_tokens budget (needed
            # for a one-line score) leaves `content` empty. Verified live.
            reasoning_effort="low",
        )
        return response.choices[0].message.content.strip()

    return _call


# Bumped whenever the rubric or prompt shape changes: a calibration result
# (FR-14) is only valid for the version it was measured against, and
# `metrics_json["judge"]["prompt_version"]` records which judge scored a case.
JUDGE_PROMPT_VERSION = "v2-banded-rubric"

# v1 ("Score how well the answer addresses the question", endpoints only)
# measured kappa=0.450 against 200 human labels — below FR-15's 0.70 — and on a
# real pipeline scored a *worse* candidate 0.22 higher, because it never saw
# what a correct answer looks like. This version fixes the three causes named
# in calibration/calibration_report.md §4: the scale's middle is defined, the
# expected answer and retrieved context are shown when available, and grading
# correctness over fluency is stated outright.
DEFAULT_RUBRIC = (
    "Score the answer from 0.0 to 1.0 using these bands:\n"
    "  1.0 - fully and correctly answers the question; matches the expected answer when one is given\n"
    "  0.8 - correct and useful, with only minor omissions or extra wording\n"
    "  0.6 - partially correct: addresses the question but is incomplete, vague, or mixes in an error\n"
    "  0.4 - on topic but largely fails to answer, or is mostly wrong\n"
    "  0.2 - wrong, or fluent and confident while unsupported by the context provided\n"
    "  0.0 - does not address the question at all\n"
    "Rules:\n"
    "- Grade correctness and usefulness, NOT fluency. A confident, well-written wrong answer scores 0.2 or below.\n"
    "- If an expected answer is given, an answer that contradicts it cannot score above 0.2, however plausible it reads.\n"
    "- If context is given, a claim absent from that context is unsupported, even if it happens to be true.\n"
    '- "I don\'t know" scores 0.8 when the context genuinely lacks the answer, and 0.2 when the context contains it.'
)

# One clear pass, one clear fail, and — deliberately — one borderline case.
# The 2026-09-20 calibration put 22 of 38 false passes in the band the judge
# had no anchor for, so the middle of the scale is the example that matters.
_FEW_SHOT = (
    "Examples:\n"
    "Question: What year did the Ford Model T go on sale? | Expected: 1908 | Answer: 1908. -> 1.0\n"
    "Question: Who directed Jaws? | Expected: Steven Spielberg | "
    "Answer: Jaws is a landmark 1975 thriller widely credited with inventing the summer blockbuster. -> 0.4\n"
    "Question: What is the battery warranty? | Expected: 3 years | "
    "Answer: Batteries are covered under the standard warranty, which is generous compared to competitors. -> 0.6\n"
)

_ABSOLUTE_PROMPT = (
    "You are grading an AI assistant's answer.\n\n"
    "Question: {query}\n"
    "{reference_block}{context_block}"
    "Answer: {answer}\n\n"
    "{rubric}\n\n"
    "{few_shot}\n"
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


def build_absolute_prompt(
    query: str,
    answer: str,
    expected: str | None = None,
    context: str | None = None,
    rubric: str = DEFAULT_RUBRIC,
    few_shot: str = _FEW_SHOT,
) -> str:
    """The graded prompt. `expected` and `context` are optional because not
    every golden case has them (a HelpSteer2 helpfulness label has neither) —
    each block is omitted rather than sent empty, so the judge is never shown
    an "Expected answer:" heading with nothing under it."""
    return _ABSOLUTE_PROMPT.format(
        query=query,
        answer=answer,
        rubric=rubric,
        few_shot=few_shot,
        reference_block=f"Expected answer: {expected}\n" if expected else "",
        context_block=f"Context the answer had to rely on:\n{context}\n" if context else "",
    )


def score_absolute(
    llm_call: LLMCall,
    query: str,
    answer: str,
    expected: str | None = None,
    context: str | None = None,
    rubric: str = DEFAULT_RUBRIC,
    few_shot: str = _FEW_SHOT,
) -> float | None:
    """Absolute rubric score in [0, 1], or `None` if the judge's response
    couldn't be parsed as a number in range — never silently coerced to 0.0,
    which would misreport "unparseable" as "worst possible score"."""
    return _parse_score(llm_call(build_absolute_prompt(query, answer, expected, context, rubric, few_shot)))


def score_absolute_with_rationale(
    llm_call: LLMCall,
    query: str,
    answer: str,
    expected: str | None = None,
    context: str | None = None,
    rubric: str = DEFAULT_RUBRIC,
    few_shot: str = _FEW_SHOT,
) -> tuple[float | None, str]:
    """Same as `score_absolute`, but also returns the judge's raw
    completion text (PRD FR-28 — dashboard drill-down wants the rationale,
    not just the parsed number). A separate function rather than changing
    `score_absolute`'s return shape, so existing callers/tests of the
    plain scorer are untouched."""
    raw = llm_call(build_absolute_prompt(query, answer, expected, context, rubric, few_shot))
    return _parse_score(raw), raw


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
