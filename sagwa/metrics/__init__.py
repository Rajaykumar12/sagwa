"""Metrics layer: reference-based (exact/fuzzy/ROUGE/embedding),
classification (precision/recall/F1 against label ground truth), RAGAS
reference-free RAG metrics, and safety flags (PRD FR-8-FR-12).

`compute_metrics()` is the single entry point `cli.py`'s `run()` calls per
case — which metrics apply is decided from the golden case's own fields
(`task_type`, `expected_output`, `expected_labels`), so adding a new task
type's metrics later doesn't require storage/CLI changes (NFR
"Extensibility").
"""
from __future__ import annotations

from sagwa.adapters.base import AgentRun
from sagwa.datasets.schema import GoldenCase
from sagwa.metrics.classification import compute_classification_metrics
from sagwa.metrics.reference import compute_reference_metrics
from sagwa.metrics.safety import compute_safety_flags


def compute_metrics(case: GoldenCase, answer: str, context: str | None) -> dict:
    metrics: dict = {"safety": compute_safety_flags(answer)}

    if case.expected_output:
        metrics["reference"] = compute_reference_metrics(answer, case.expected_output)

    if case.expected_labels:
        metrics["classification"] = compute_classification_metrics(answer, case.expected_labels)

    if case.task_type == "rag_qa" and context:
        from sagwa.metrics.ragas_metrics import compute_ragas_metrics

        metrics["ragas"] = compute_ragas_metrics(query=case.input, context=context, answer=answer)

    from sagwa.metrics.judge_metrics import compute_judge_metric

    judge = compute_judge_metric(
        query=case.input, answer=answer, expected=case.expected_output, context=context
    )
    if judge is not None:
        metrics["judge"] = judge

    return metrics


def compute_agent_metrics(case: GoldenCase, run: AgentRun) -> dict:
    """Metrics only an agentic case has (docs/AGENTIC_PRD.md Phase 2):
    `task.success` when the case defines success criteria, and `agent.*` when
    the run recorded a trajectory. Kept apart from `compute_metrics`, whose
    callers pass only an answer and context. Pure functions, no model call,
    so a stored run can be rescored from its trajectory at any time."""
    from sagwa.metrics.agent_success import compute_success
    from sagwa.metrics.agent_trajectory import compute_trajectory_metrics

    metrics: dict = {}
    if case.success_criteria is not None:
        metrics["task"] = compute_success(case.success_criteria, run)
    if run.trajectory:
        metrics["agent"] = compute_trajectory_metrics(run, case.expected_tools)
    return metrics
