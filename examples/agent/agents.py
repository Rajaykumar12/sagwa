"""Where the four agents' prompts come from, and the one Groq call they all go through.

Each agent is a prompted call, not a class: planner -> researcher (the only
one with tools) -> writer -> reviewer. Prompts are short on purpose: on
Groq's free tier a token spent on a prompt is a token not spent on a case.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from sagwa.prompts import PromptStore

MODEL = os.environ.get("SAGWA_AGENT_MODEL", "openai/gpt-oss-20b")

# The four agents' prompts live in the versioned prompt store, not here:
# examples/agent/prompts/<planner|researcher|writer|reviewer|tool_descriptions>/vN.txt.
# `sagwa improve` adds versions there; a variant pins the versions it uses.
STORE = PromptStore(Path(__file__).parent / "prompts")
PROMPT_NAMES = ("planner", "researcher", "writer", "reviewer", "tool_descriptions")


def chat(client, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 700):
    """One completion. Returns (message, tokens, latency_ms)."""
    start = time.perf_counter()
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens,
        reasoning_effort="low", **kwargs,
    )
    return (response.choices[0].message, response.usage.total_tokens,
            int((time.perf_counter() - start) * 1000))
