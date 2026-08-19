"""Shared case-level pass/fail predicate, used by `sagwa.diff` (FR-17's
flip list) and `sagwa.clustering` (FR-20's failing-case input) alike —
factored out here rather than clustering importing from diff, or each
module reimplementing it.

Lives under `sagwa.gate` because the definition is inseparable from
`config/gates.yaml`'s `{op, value}` comparison (`sagwa.gate._compare`),
which is gate's own vocabulary.
"""
from __future__ import annotations

from sagwa.gate import _compare
from sagwa.storage import Result


def _get_metric(result: Result, dotted_path: str):
    value = result.metrics_json or {}
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def case_passes(result: Result, gates: dict) -> bool | None:
    """Whether `result` passes every metric configured in `gates` (the
    `config/gates.yaml` shape: `{metric_name: {op, value}}`, where
    `metric_name` is a dotted `metrics_json` path). Returns `None` — not
    False — when the case errored or none of the configured metrics were
    computed for it, since "no data" isn't the same claim as "failed"."""
    if result.error is not None:
        return None
    if not gates:
        return None

    saw_any = False
    for metric_name, rule in gates.items():
        observed = _get_metric(result, metric_name)
        if observed is None:
            continue
        saw_any = True
        if not _compare(float(observed), rule["op"], float(rule["value"])):
            return False
    return True if saw_any else None
