"""CI gate (PRD FR-23-FR-25): evaluates one run's aggregate per-metric
values against thresholds defined in a version-controlled config
(`config/gates.yaml`, FR-25), and fails (non-zero exit, via `sagwa/cli.py`)
rather than silently passing when a threshold isn't met.

Scope, per FR-23's literal text ("thresholds configured per metric"): this
gates one run's absolute metric values, not a diff against a baseline. A
metric configured in the gate but never computed for the run (e.g. RAGAS
degraded to `None`) fails loudly — `observed=None`, `passed=False` — rather
than being silently skipped, matching `sagwa/judge/calibration.py`'s
"refuse rather than degrade" philosophy for gating decisions.

The same `config/gates.yaml` thresholds also define case-level pass/fail
for `sagwa/diff` and `sagwa/clustering` (see `sagwa.diff.case_passes`) —
this file is the one place the `{op, value}` comparison itself lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sagwa.storage import Result

_OPS = {
    "gt": lambda observed, threshold: observed > threshold,
    "gte": lambda observed, threshold: observed >= threshold,
    "lt": lambda observed, threshold: observed < threshold,
    "lte": lambda observed, threshold: observed <= threshold,
}


class GateConfigError(Exception):
    """Missing or malformed `config/gates.yaml` — fails loudly rather than
    silently gating on nothing (PRD FR-25)."""


@dataclass
class MetricGateResult:
    metric_name: str
    op: str
    threshold: float
    observed: float | None
    passed: bool

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "op": self.op,
            "threshold": self.threshold,
            "observed": self.observed,
            "passed": self.passed,
        }


@dataclass
class GateResult:
    run_id: str
    metric_results: list[MetricGateResult]
    passed: bool
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "metric_results": [m.to_dict() for m in self.metric_results],
            "passed": self.passed,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """PR-comment-ready summary (FR-24)."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [f"## Gate result for run `{self.run_id}`: {status}", "", "| metric | op | threshold | observed | passed |", "|---|---|---|---|---|"]
        for m in self.metric_results:
            observed = f"{m.observed:.3f}" if m.observed is not None else "n/a"
            lines.append(f"| {m.metric_name} | {m.op} | {m.threshold} | {observed} | {'✅' if m.passed else '❌'} |")
        return "\n".join(lines)


def load_gate_config(path: Path | str = Path("config/gates.yaml")) -> dict:
    path = Path(path)
    if not path.exists():
        raise GateConfigError(f"{path}: gate config not found")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise GateConfigError(f"{path}: invalid YAML ({e})") from e

    metrics = (raw or {}).get("metrics")
    if not metrics:
        raise GateConfigError(f"{path}: no 'metrics' section defined")
    for metric_name, rule in metrics.items():
        if "op" not in rule or rule["op"] not in _OPS:
            raise GateConfigError(f"{path}: metric '{metric_name}' has an invalid or missing 'op'")
        if "value" not in rule:
            raise GateConfigError(f"{path}: metric '{metric_name}' is missing 'value'")
    return metrics


def _compare(observed: float, op: str, threshold: float) -> bool:
    if op not in _OPS:
        raise ValueError(f"unknown gate op '{op}' — expected one of {sorted(_OPS)}")
    return _OPS[op](observed, threshold)


def _get_metric(result: Result, dotted_path: str):
    value = result.metrics_json or {}
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def aggregate_metric(results: list[Result], metric_path: str) -> float | None:
    """Mean of `metric_path` (a dotted `metrics_json` lookup, e.g.
    'ragas.faithfulness') over the run's cases, excluding errored cases
    (which never got metrics computed) and cases where the metric is
    absent. Returns `None` — not 0.0 — when no case has the metric."""
    values = []
    for result in results:
        if result.error is not None:
            continue
        value = _get_metric(result, metric_path)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_gate(session, run_id: str, config: dict) -> GateResult:
    """Entry point (FR-23). `config` is `load_gate_config()`'s return value:
    `{metric_name: {op, value}}`."""
    results = session.query(Result).filter(Result.run_id == run_id).all()

    metric_results = []
    for metric_name, rule in config.items():
        observed = aggregate_metric(results, metric_name)
        passed = observed is not None and _compare(observed, rule["op"], float(rule["value"]))
        metric_results.append(
            MetricGateResult(
                metric_name=metric_name,
                op=rule["op"],
                threshold=float(rule["value"]),
                observed=observed,
                passed=passed,
            )
        )

    return GateResult(
        run_id=run_id,
        metric_results=metric_results,
        passed=all(m.passed for m in metric_results),
    )
