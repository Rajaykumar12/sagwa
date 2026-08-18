"""Metrics layer: reference-based (exact/fuzzy/ROUGE/embedding), RAGAS
reference-free RAG metrics, and safety flags (PRD FR-8-FR-12).

`compute_metrics()` is the single entry point `cli.py`'s `run()` calls per
case — which metrics apply is decided from the golden case's own fields
(`task_type`, `expected_output`), so adding a new task type's metrics
later doesn't require storage/CLI changes (NFR "Extensibility").
"""
from __future__ import annotations

from sagwa.datasets.schema import GoldenCase
from sagwa.metrics.reference import compute_reference_metrics
from sagwa.metrics.safety import compute_safety_flags


def compute_metrics(case: GoldenCase, answer: str, context: str | None) -> dict:
    metrics: dict = {"safety": compute_safety_flags(answer)}

    if case.expected_output:
        metrics["reference"] = compute_reference_metrics(answer, case.expected_output)

    if case.task_type == "rag_qa" and context:
        from sagwa.metrics.ragas_metrics import compute_ragas_metrics

        metrics["ragas"] = compute_ragas_metrics(query=case.input, context=context, answer=answer)

    return metrics
