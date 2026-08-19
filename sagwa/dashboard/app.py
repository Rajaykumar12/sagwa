"""Streamlit dashboard (PRD FR-26..FR-28) — rendering only; all the actual
querying/aggregation lives in `queries.py` (unit-tested independently).
Launched via `sagwa dashboard` (see `sagwa/cli.py`), or directly with
`streamlit run sagwa/dashboard/app.py`.
"""
import streamlit as st

from sagwa.dashboard.queries import case_detail, cost_latency_trend, metric_trend, run_failure_clusters
from sagwa.gate import load_gate_config
from sagwa.storage import Run, get_session

st.set_page_config(page_title="Sagwa", layout="wide")
st.title("Sagwa: eval run dashboard")

with get_session() as session:
    target_names = sorted({row[0] for row in session.query(Run.target_name).distinct()})

if not target_names:
    st.info("No runs recorded yet. Run `sagwa run --target ... --dataset ...` first.")
    st.stop()

target_name = st.selectbox("Target pipeline", target_names)

st.header("Metric trends")
metric_path = st.text_input("Metric path (dotted, e.g. 'reference.fuzzy_match' or 'judge.score')", "reference.fuzzy_match")
with get_session() as session:
    points = metric_trend(session, target_name, metric_path)
if points:
    st.line_chart({"created_at": [p[0] for p in points], metric_path: [p[1] for p in points]}, x="created_at", y=metric_path)
else:
    st.write(f"No runs of '{target_name}' have any case reporting '{metric_path}'.")

st.header("Cost & latency trends")
with get_session() as session:
    cl_points = cost_latency_trend(session, target_name)
if cl_points:
    st.line_chart(
        {
            "created_at": [p[0] for p in cl_points],
            "mean_cost_usd": [p[1] for p in cl_points],
            "mean_latency_ms": [p[2] for p in cl_points],
        },
        x="created_at",
    )

st.header("Failure clusters")
with get_session() as session:
    run_ids = [
        r.id
        for r in session.query(Run).filter(Run.target_name == target_name).order_by(Run.created_at.desc()).all()
    ]
if run_ids:
    run_id = st.selectbox("Run", run_ids)
    gates = load_gate_config() if st.checkbox("Use config/gates.yaml thresholds", value=True) else {}
    with get_session() as session:
        clusters = run_failure_clusters(session, run_id, gates)

    if not clusters:
        st.write("No failing cases for this run (per the current pass/fail config).")
    for c in sorted(clusters, key=lambda c: c.size, reverse=True):
        with st.expander(f"[{c.cluster_id}] {c.label} — {c.size} case(s)"):
            for case_id in c.case_ids:
                with get_session() as session:
                    detail = case_detail(session, run_id, case_id)
                if detail is None:
                    continue
                st.markdown(f"**{case_id}**")
                st.text(f"input: {detail['input']}")
                st.text(f"output: {detail['output']}")
                if detail["judge_rationale"]:
                    st.text(f"judge score: {detail['judge_score']}  |  rationale: {detail['judge_rationale']}")
