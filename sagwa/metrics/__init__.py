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

    judge = compute_judge_metric(query=case.input, answer=answer)
    if judge is not None:
        metrics["judge"] = judge

    return metrics
