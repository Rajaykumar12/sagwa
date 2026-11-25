"""The agent loop: plan, research with tools, hand off, write, verify.
Hand-rolled on Groq with no framework (locked decision #6), and every step it
takes is recorded in the returned `AgentRun`'s trajectory.

    python -m examples.agent.orchestrator "How many orders shipped late last quarter?"
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

from examples.agent import agents
from examples.agent.tools import calculator, corpus_search, order_db, shipment_tracker, tool_schemas
from examples.agent.world import build_database
from sagwa.adapters.base import AgentRun, AgentStep


@dataclass
class AgentConfig:
    """Every knob a regression variant turns. The baseline is the default."""

    # Prompt store version per prompt (see agents.PROMPT_NAMES); a prompt not
    # listed uses the store's `current` version.
    prompt_versions: dict[str, str] = field(default_factory=dict)
    # Name the model sees for a tool, when different from its real name.
    tool_names: dict[str, str] = field(default_factory=dict)
    # Consecutive failures of one tool before the researcher is told to stop
    # calling it. None = never (V3_NoRetryCap).
    retry_cap: int | None = 2
    # Characters of the researcher's findings passed to the writer. None = all.
    handoff_chars: int | None = None
    # Hard stop on trajectory length, in every variant, so even a runaway agent
    # ends. It's a safety net, not the retry cap; golden cases allow at most 20
    # steps, so an agent that only stops here has already failed.
    max_steps: int = 24


def _groq():
    from groq import Groq

    return Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6, timeout=90.0)


class Orchestrator:
    def __init__(self, config: AgentConfig | None = None, client=None):
        self.config = config or AgentConfig()
        self.client = client or _groq()
        self.conn = build_database()
        # Resolved once, so every case in a run uses the same versions.
        self.prompt_versions = {name: self.config.prompt_versions.get(name) or agents.STORE.current(name)
                                for name in agents.PROMPT_NAMES}
        self.prompts = {name: agents.STORE.load(name, v) for name, v in self.prompt_versions.items()}
        self.tools = tool_schemas(json.loads(self.prompts["tool_descriptions"]), self.config.tool_names)
        self.real_name = {alias: name for name, alias in self.config.tool_names.items()}

    def _call_tool(self, name: str, args: dict) -> str:
        if name == "corpus_search":
            return corpus_search(args["query"])
        if name == "order_db":
            return order_db(self.conn, args["sql"])
        if name == "calculator":
            return calculator(args["expression"])
        if name == "shipment_tracker":
            return shipment_tracker(args["order_id"])
        raise ValueError(f"unknown tool: {name}")

    def _agent_step(self, steps, name, prompt, max_tokens=700) -> str:
        message, tokens, latency = agents.chat(self.client, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
        output = (message.content or "").strip()
        steps.append(AgentStep("agent", name, input=prompt, output=output, latency_ms=latency,
                               tokens=tokens, metadata={"model": agents.MODEL}))
        return output

    def _research(self, steps: list[AgentStep], question: str, plan: str) -> str:
        messages = [{"role": "system", "content": self.prompts["researcher"]},
                    {"role": "user", "content": f"Question: {question}\n\nPlan:\n{plan}"}]
        failures: dict[str, int] = {}
        while len(steps) < self.config.max_steps:
            message, tokens, latency = agents.chat(self.client, messages, tools=self.tools)
            calls = message.tool_calls or []
            steps.append(AgentStep("agent", "researcher", input=messages[-1]["content"],
                                   output=(message.content or "").strip() or None, latency_ms=latency,
                                   tokens=tokens, metadata={"model": agents.MODEL, "tool_calls": len(calls)}))
            if not calls:
                return (message.content or "").strip()
            messages.append({"role": "assistant", "content": message.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for call in calls:
                name = self.real_name.get(call.function.name, call.function.name)
                start = time.perf_counter()
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output, error = self._call_tool(name, args), None
                    failures[name] = 0
                except Exception as e:  # noqa: BLE001 — a failed tool call is data, not a crash
                    args, output, error = {"raw": call.function.arguments}, None, f"{type(e).__name__}: {e}"
                    failures[name] = failures.get(name, 0) + 1
                steps.append(AgentStep("tool", name, input=json.dumps(args), output=output,
                                       latency_ms=int((time.perf_counter() - start) * 1000), error=error,
                                       metadata={"arguments": args}))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": output if error is None else f"ERROR: {error}"})
            capped = [n for n, k in failures.items() if self.config.retry_cap and k >= self.config.retry_cap]
            if capped:
                messages.append({"role": "user", "content":
                                 f"The {', '.join(capped)} tool failed {self.config.retry_cap} times in a row. "
                                 "Stop calling it. Report your findings now, including what you could not find."})
                failures = {n: k for n, k in failures.items() if n not in capped}
        return "Research stopped at the step limit before finishing."

    def run(self, question: str) -> AgentRun:
        start = time.perf_counter()
        steps: list[AgentStep] = []
        plan = self._agent_step(steps, "planner", self.prompts["planner"].format(question=question), max_tokens=400)
        findings = self._research(steps, question, plan)

        payload = findings if self.config.handoff_chars is None else findings[: self.config.handoff_chars]
        steps.append(AgentStep("handoff", "researcher->writer", input=findings, output=payload,
                               metadata={"truncated": len(payload) < len(findings)}))

        answer = self._agent_step(steps, "writer", self.prompts["writer"].format(question=question, findings=payload, feedback=""))
        # The reviewer runs only when the plan asked for verification, so a
        # planner that stops planning it (V4) really does skip validation.
        if "verify" in plan.lower():
            verdict = self._agent_step(steps, "reviewer", self.prompts["reviewer"].format(
                question=question, findings=payload, answer=answer), max_tokens=300)
            if verdict.upper().startswith("REVISE"):
                answer = self._agent_step(steps, "writer", self.prompts["writer"].format(
                    question=question, findings=payload, feedback=f"\n\nReviewer feedback on your draft: {verdict}"))

        return AgentRun(
            answer=answer, context=payload,
            latency_ms=int((time.perf_counter() - start) * 1000),
            tokens=sum(s.tokens or 0 for s in steps),
            cost_usd=None,  # free tier: tokens are the honest cost measure here
            trajectory=steps,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")
    run = Orchestrator().run(" ".join(sys.argv[1:]) or "How many orders shipped late last quarter?")
    for i, s in enumerate(run.trajectory):
        detail = s.error or (s.output or "")
        print(f"{i:2d} {s.step_type:7s} {s.name:20s} {s.tokens or '':>5} {detail[:110]!r}")
    print(f"\nANSWER: {run.answer}\n({run.step_count} steps, {run.tokens} tokens, {run.latency_ms} ms)")
