"""Agent trace view (docs/AGENTIC_PRD.md Phase 4): a case's trajectory as a
vertical timeline, plus a drill-down into any step.

`trace_dot` is a pure function (DOT text for `st.graphviz_chart`, which
renders in the browser: no Graphviz install), so the picture is testable
without a Streamlit runtime. `render_trace` only lays it out.
"""
from __future__ import annotations

import json

_SHAPE = {"agent": "box", "tool": "ellipse", "handoff": "cds"}
_FILL = {"agent": "#dbe8f6", "tool": "#e4f2e1", "handoff": "#f4ecd8"}
_FAILED_FILL = "#f6d5d5"


def _label(step: dict) -> str:
    parts = [f"{step['seq']}. {step['name']}"]
    detail = []
    if step["latency_ms"] is not None:
        detail.append(f"{step['latency_ms']} ms")
    if step["tokens"]:
        detail.append(f"{step['tokens']} tok")
    if detail:
        parts.append(" · ".join(detail))
    if step["error"]:
        parts.append("FAILED: " + step["error"][:40])
    return "\\n".join(p.replace('"', "'") for p in parts)


def trace_dot(steps: list[dict]) -> str:
    """DOT for the trajectory, top to bottom. A failed step is filled red, so
    retries and loops stand out as runs of red nodes."""
    lines = ["digraph trace {", "  rankdir=TB; nodesep=0.15; ranksep=0.2;",
             '  node [fontname="Helvetica", fontsize=9, style=filled, height=0.3, margin="0.1,0.03"];',
             "  edge [arrowsize=0.5];"]
    for s in steps:
        fill = _FAILED_FILL if s["error"] else _FILL.get(s["step_type"], "#eeeeee")
        lines.append(f'  s{s["seq"]} [label="{_label(s)}", shape={_SHAPE.get(s["step_type"], "box")}, '
                     f'fillcolor="{fill}"];')
    for a, b in zip(steps, steps[1:]):
        lines.append(f"  s{a['seq']} -> s{b['seq']};")
    lines.append("}")
    return "\n".join(lines)


def render_trace(st, steps: list[dict], summary: list[dict], case: dict) -> None:
    """Timeline, roll-up and step drill-down for one case. `case` is the
    `queries.case_detail` dict for the same result."""
    if not steps:
        st.info("This case recorded no trajectory.")
        return

    task = (case.get("metrics_json") or {}).get("task") or {}
    if task.get("success") is False:
        st.error(f"task.success = False — failed conditions: {', '.join(task.get('failed', [])) or 'n/a'}")
    elif task.get("success"):
        st.success("task.success = True")
    failed_steps = [s for s in steps if s["error"]]
    if failed_steps:
        st.warning(f"{len(failed_steps)} of {len(steps)} steps failed: "
                   + ", ".join(f"#{s['seq']} {s['name']}" for s in failed_steps[:8]))

    timeline, detail = st.columns([1, 1])
    with timeline:
        # Natural size: stretched to the column, a 10-step trace runs screens tall.
        st.graphviz_chart(trace_dot(steps), use_container_width=False)
    with detail:
        st.markdown("**Answer**")
        st.text(case.get("output") or "")
        st.markdown("**Per-agent / per-tool roll-up**")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        agent_metrics = (case.get("metrics_json") or {}).get("agent")
        if agent_metrics:
            st.markdown("**Trajectory metrics**")
            st.json(agent_metrics, expanded=False)

        by_seq = {s["seq"]: s for s in steps}
        seq = st.selectbox("Step", list(by_seq),
                           format_func=lambda i: f"{i}. {by_seq[i]['step_type']} · {by_seq[i]['name']}"
                           + ("  ✗" if by_seq[i]["error"] else ""))
        step = by_seq[seq]
        meta = step["metadata"]
        st.markdown(
            f"**{step['step_type']} · {step['name']}**  \n"
            f"model: `{meta.get('model', 'n/a')}` · prompt version: `{meta.get('prompt_version', 'n/a')}`  \n"
            f"latency: {step['latency_ms'] if step['latency_ms'] is not None else 'n/a'} ms · "
            f"tokens: {step['tokens'] or 'n/a'} · cost: {step['cost_usd'] if step['cost_usd'] is not None else 'n/a'}"
        )
        if step["error"]:
            st.error(step["error"])
        if "arguments" in meta:
            st.markdown("**Tool arguments**")
            st.code(json.dumps(meta["arguments"], indent=2), language="json")
        with st.expander("Input", expanded=step["step_type"] == "tool"):
            st.text(step["input"] or "")
        with st.expander("Output", expanded=True):
            st.text(step["output"] or "")
