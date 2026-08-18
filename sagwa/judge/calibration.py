"""Judge calibration workflow (PRD FR-13-FR-15a).

Given judge scores/labels and corresponding human labels, computes
agreement (accuracy, Cohen's kappa) and a confusion matrix; stores the
result as a versioned artifact tied to a judge prompt version (FR-14);
refuses to let an uncalibrated (or under-threshold) judge be used for
gating (FR-15); and supports scoring a prior/external judge — e.g. ringo's
`backend/eval.py` — against the same human-labeled set for a real
before/after comparison (FR-15a).

The 150-200 real human labels this is meant to run against are not
something this module can generate — see docs/GAP_ANALYSIS.md. Tests for
this module use a small, explicitly-synthetic fixture to validate the math
and the artifact/refusal plumbing, not to claim a real calibration result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CalibrationResult:
    judge_prompt_version: str
    n: int
    accuracy: float
    cohen_kappa: float
    confusion_matrix: dict  # {"human_label|judge_label": count}
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    baseline_name: str | None = None  # e.g. "ringo_eval_py" for FR-15a comparisons

    def to_dict(self) -> dict:
        return {
            "judge_prompt_version": self.judge_prompt_version,
            "n": self.n,
            "accuracy": self.accuracy,
            "cohen_kappa": self.cohen_kappa,
            "confusion_matrix": self.confusion_matrix,
            "created_at": self.created_at,
            "baseline_name": self.baseline_name,
        }


def cohen_kappa(human_labels: list, judge_labels: list) -> float:
    """Cohen's kappa, hand-computed (no scikit-learn dependency needed for
    a formula this direct): (observed agreement - expected agreement) /
    (1 - expected agreement)."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must be the same length")
    n = len(human_labels)
    if n == 0:
        raise ValueError("cannot compute kappa on an empty label set")

    categories = sorted(set(human_labels) | set(judge_labels), key=str)
    human_counts = {c: 0 for c in categories}
    judge_counts = {c: 0 for c in categories}
    agree = 0
    for h, j in zip(human_labels, judge_labels):
        human_counts[h] += 1
        judge_counts[j] += 1
        if h == j:
            agree += 1

    observed_agreement = agree / n
    expected_agreement = sum((human_counts[c] / n) * (judge_counts[c] / n) for c in categories)
    if expected_agreement >= 1:
        return 1.0  # no disagreement was possible; avoid a divide-by-zero
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def confusion_matrix(human_labels: list, judge_labels: list) -> dict:
    """{"human_label|judge_label": count} — string-keyed so it round-trips
    through JSON regardless of the label type."""
    matrix: dict[str, int] = {}
    for h, j in zip(human_labels, judge_labels):
        key = f"{h}|{j}"
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


def calibrate(
    human_labels: list,
    judge_labels: list,
    judge_prompt_version: str,
    baseline_name: str | None = None,
) -> CalibrationResult:
    """Agreement between `judge_labels` and ground-truth `human_labels`
    (PRD FR-13). Pass `baseline_name` when scoring a prior judge (e.g.
    ringo's eval.py) against the same set, for the FR-15a comparison."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must be the same length")
    n = len(human_labels)
    if n == 0:
        raise ValueError("cannot calibrate on an empty label set")

    accuracy = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n
    return CalibrationResult(
        judge_prompt_version=judge_prompt_version,
        n=n,
        accuracy=accuracy,
        cohen_kappa=cohen_kappa(human_labels, judge_labels),
        confusion_matrix=confusion_matrix(human_labels, judge_labels),
        baseline_name=baseline_name,
    )


def save_calibration_result(result: CalibrationResult, directory: Path | str = Path("calibration")) -> Path:
    """Persists one calibration run as a versioned JSON artifact (PRD
    FR-14). Filename includes the prompt version, baseline name (if any),
    and timestamp — a judge-prompt change produces a new, distinguishable
    record rather than overwriting history (append-only, matching the
    Run/Result storage convention)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"_{result.baseline_name}" if result.baseline_name else ""
    timestamp = result.created_at.replace(":", "").replace("+00:00", "Z")
    path = directory / f"calibration_{result.judge_prompt_version}{suffix}_{timestamp}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def load_calibration_results(directory: Path | str = Path("calibration")) -> list[CalibrationResult]:
    directory = Path(directory)
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("calibration_*.json")):
        results.append(CalibrationResult(**json.loads(path.read_text())))
    return results


class CalibrationRequired(Exception):
    """Raised when a judge has no recorded calibration above the required
    kappa threshold — refuses to let an uncalibrated judge gate a build
    (PRD FR-15)."""


def require_calibration(
    judge_prompt_version: str,
    min_kappa: float = 0.70,
    directory: Path | str = Path("calibration"),
) -> CalibrationResult:
    """Returns the most recent calibration record for `judge_prompt_version`
    that meets `min_kappa`, or raises `CalibrationRequired`. Call this
    before using a judge for any gating decision (PRD FR-15) — never skip
    it "just this once"."""
    matching = [
        r
        for r in load_calibration_results(directory)
        if r.judge_prompt_version == judge_prompt_version and r.cohen_kappa >= min_kappa
    ]
    if not matching:
        raise CalibrationRequired(
            f"No calibration result for judge prompt version '{judge_prompt_version}' "
            f"reaches kappa >= {min_kappa}. Run the calibration workflow before using "
            "this judge for gating (PRD FR-15)."
        )
    return max(matching, key=lambda r: r.created_at)
