"""The one place a metric path is declared (agentic PRD Phase 0).

`sagwa diff` used to keep its own hardcoded tuples of metric paths, separate
from the code that computes them. Twice a metric existed but was missing from
those tuples, so diff silently dropped it: classification metrics, then
faithfulness via NaN handling (docs/STATUS.md §5). The agentic phases add
about twenty metrics, so every consumer now reads from this registry instead.

`kind` picks the significance test in `sagwa.diff` (paired bootstrap CI for
continuous, exact McNemar's for binary). `aggregation` picks how
`sagwa.gate.aggregate_metric` rolls a run up: "mean" over cases, or
"run_level" for a metric that is not a mean of per-case values, such as cost
per successful task (total cost / successful cases). A run-level metric has no
per-case values, so diff can't run a paired test on it and skips it.

`higher_is_better` is declared here for metrics, but the gate still takes
direction from each rule's `op` in gates.yaml, so the two can't disagree for a
configured gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from sagwa.storage import Result


@dataclass(frozen=True)
class MetricSpec:
    kind: Literal["continuous", "binary"]
    higher_is_better: bool
    aggregation: Literal["mean", "run_level"] = "mean"
    # Required when aggregation == "run_level": all of a run's results -> value.
    aggregator: Callable[[list["Result"]], float | None] | None = None


# Order is significant: `sagwa diff` reports metrics in this order, continuous
# first, so its output stays identical to before the registry existed.
METRICS: dict[str, MetricSpec] = {
    "reference.fuzzy_match": MetricSpec("continuous", higher_is_better=True),
    "reference.rouge_l_f1": MetricSpec("continuous", higher_is_better=True),
    "reference.embedding_similarity": MetricSpec("continuous", higher_is_better=True),
    "classification.precision": MetricSpec("continuous", higher_is_better=True),
    "classification.recall": MetricSpec("continuous", higher_is_better=True),
    "classification.f1": MetricSpec("continuous", higher_is_better=True),
    "ragas.faithfulness": MetricSpec("continuous", higher_is_better=True),
    "ragas.context_precision": MetricSpec("continuous", higher_is_better=True),
    "judge.score": MetricSpec("continuous", higher_is_better=True),
    "reference.exact_match": MetricSpec("binary", higher_is_better=True),
    "classification.exact_set_match": MetricSpec("binary", higher_is_better=True),
    "safety.pii.flagged": MetricSpec("binary", higher_is_better=False),
    "safety.toxicity.flagged": MetricSpec("binary", higher_is_better=False),
}


def metric_paths(kind: Literal["continuous", "binary"]) -> tuple[str, ...]:
    """Per-case (mean-aggregated) metric paths of one kind, in registry order."""
    return tuple(
        path for path, spec in METRICS.items() if spec.kind == kind and spec.aggregation == "mean"
    )
