# Using Sagwa from an agent (MCP)

`sagwa mcp` serves Sagwa's evaluation tools over the [Model Context Protocol](https://modelcontextprotocol.io), on stdio. A coding agent can then answer "did my change regress the pipeline?" itself: run the golden set, compare with a baseline, check the gate, and look at what failed, all through tool calls.

Each tool calls the same function as the matching CLI command and returns that result's own JSON, so an agent and a person running the CLI always get the same answer.

## Install

```bash
pip install -e ".[mcp,metrics]"     # the mcp extra needs mcp>=2.0
sagwa migrate                       # the server reads and writes the run database
```

## Register it with a client

The server runs as a subprocess of the client. Two things matter: the **working directory**, because dataset and gate-config paths are resolved against it the way the CLI resolves them, and **`DATABASE_URL`**, which names the run history.

**Claude Code**, from the repository whose pipeline you evaluate:

```bash
claude mcp add sagwa --env DATABASE_URL=sqlite:///$PWD/sagwa.db -- $PWD/.venv/bin/sagwa mcp
```

**Any client that takes a JSON config** (Claude Desktop, IDEs):

```json
{
  "mcpServers": {
    "sagwa": {
      "command": "/absolute/path/to/.venv/bin/sagwa",
      "args": ["mcp"],
      "cwd": "/absolute/path/to/your/repo",
      "env": {"DATABASE_URL": "sqlite:////absolute/path/to/your/repo/sagwa.db", "GROQ_API_KEY": "..."}
    }
  }
}
```

`GROQ_API_KEY` is only needed for the judge and RAGAS metrics. Without it, or with it empty, those metrics are skipped and no LLM call is made.

## Tools

| Tool | Same as | Returns |
|---|---|---|
| `run_evaluation(target, dataset, concurrency=2)` | `sagwa run` | `run_id`, `cases`, `errored`. Calls the target, so it can take minutes and spend tokens |
| `get_run(run_id)` | — | Target, model, dataset, prompt versions, status, case and error counts, agent successes |
| `compare_runs(baseline_run_id, candidate_run_id, gates_config)` | `sagwa diff` | Per-metric deltas with significance, per-tag deltas, flipped cases, most affected tag, and the CLI table as `table` |
| `check_gate(run_id, gates_config, baseline_run_id=None)` | `sagwa gate` | `passed`, each threshold, and each regression against the baseline |
| `inspect_failure(run_id, case_id)` | dashboard drill-down | Input, output, context, error, every metric with the judge's rationale, and a trajectory roll-up |
| `get_trace(run_id, case_id)` | dashboard trace view | An agent case's steps: type, name, input, output, latency, tokens, error, tool arguments |
| `cluster_failures(run_id, gates_config, min_cluster_size=3)` | `sagwa cluster` | Failing cases grouped by similarity, with labels |
| `root_causes(run_id, baseline_run_id=None)` | `sagwa rootcause` | Agent failures grouped by cause (retry loop, scope, handoff, wrong tool, skipped step, ...) with evidence, plus the text report |

An unknown run id, a missing case or file, or a bad gate config comes back as a tool error whose message says which. Other failures stay on the server's log.

## Worked example: "did my change regress the pipeline?"

An agent has just edited a prompt in the pipeline behind `my_pkg.adapters:SupportAgent`. Main's last run is `b1` (from CI or an earlier `sagwa run`). The agent's tool calls, with **illustrative** results:

1. `run_evaluation(target="my_pkg.adapters:SupportAgent", dataset="golden_sets/support.jsonl")` returns `{"run_id": "c7", "cases": 40, "errored": 0}`.
2. `check_gate(run_id="c7", gates_config="config/gates.agentic.yaml", baseline_run_id="b1")` returns `passed: false`: `task.success vs baseline` dropped 0.15, significant.
3. `compare_runs(baseline_run_id="b1", candidate_run_id="c7", gates_config="config/gates.agentic.yaml")` lists 6 cases that flipped from pass to fail, with `most_affected_tags: ["policy_lookup"]`.
4. `root_causes(run_id="c7", baseline_run_id="b1")` returns `TOOL_SELECTION, 6 cases`, with evidence like "used order_db; expected corpus_search".
5. `get_trace(run_id="c7", case_id="agt-016")` shows the step where it picked the wrong tool.

The agent can now say what broke and where before anyone reviews the PR. Steps 2–5 make no model calls; only step 1 runs the pipeline.
