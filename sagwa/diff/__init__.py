"""Diff engine (PRD FR-16-FR-19): compares two runs of the same golden set.

Rows are joined on `Result.case_id` (stable across runs — see
`GoldenCase.id`'s docstring). A case present in only one run is surfaced
in `baseline_only_case_ids`/`candidate_only_case_ids`, never silently
dropped — a shrinking/growing dataset between baseline and candidate is
itself worth knowing about. Aggregate deltas are never reported as a bare
percentage (FR-18): continuous metrics get a paired bootstrap CI on the
mean delta, binary metrics (and the derived pass/fail flag) get an exact
(binomial) McNemar's test — the chi-square approximation is unreliable
under ~25 discordant pairs, which is plausible on small golden sets.

Pass/fail (needed for FR-17's "flipped pass/fail" list) is defined against
`config/gates.yaml`: a case passes if every metric present in that config
and present in the case's `metrics_json` meets its threshold, and the case
has no `error`. This makes `gates.yaml` double as the shared pass/fail
definition diff, gate, and clustering all use — not gate-only config.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sagwa.gate._predicate import _get_metric, case_passes  # re-exported: sagwa.diff.case_passes
from sagwa.metrics.registry import METRICS, metric_paths, run_level_paths
from sagwa.storage import Result

__all__ = [
    "MetricDelta",
    "CaseFlip",
    "DiffResult",
    "load_run_results",
    "case_passes",
    "bootstrap_ci",
    "mcnemar_exact",
    "diff_runs",
    "format_table",
]

# Derived from the registry, never listed here: a metric declared in
# sagwa/metrics/registry.py is diffed, and one that isn't, isn't.
_CONTINUOUS_METRIC_PATHS = metric_paths("continuous")
_BINARY_METRIC_PATHS = metric_paths("binary")


@dataclass
class MetricDelta:
    metric_name: str
    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None
    test: str  # "bootstrap_ci" | "mcnemar" | "run_level" | "insufficient_data"
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool = False
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "test": self.test,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "n": self.n,
        }


@dataclass
class CaseFlip:
    case_id: str
    direction: str  # "pass_to_fail" | "fail_to_pass"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "direction": self.direction, "tags": self.tags}


@dataclass
class DiffResult:
    baseline_run_id: str
    candidate_run_id: str
    overall: list[MetricDelta]
    by_tag: dict[str, list[MetricDelta]]
    flips: list[CaseFlip]
    baseline_only_case_ids: list[str]
    candidate_only_case_ids: list[str]
    # Tag(s) with the most pass->fail flips: where a regression concentrated.
    most_affected_tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "overall": [m.to_dict() for m in self.overall],
            "by_tag": {tag: [m.to_dict() for m in deltas] for tag, deltas in self.by_tag.items()},
            "flips": [f.to_dict() for f in self.flips],
            "baseline_only_case_ids": self.baseline_only_case_ids,
            "candidate_only_case_ids": self.candidate_only_case_ids,
            "most_affected_tags": self.most_affected_tags,
            "created_at": self.created_at,
        }


def load_run_results(session, run_id: str) -> dict[str, Result]:
    """`case_id -> Result` for one run, one query."""
    rows = session.query(Result).filter(Result.run_id == run_id).all()
    return {row.case_id: row for row in rows}


# Below this many paired cases the bootstrap never calls a change significant:
# resampling 2 or 3 differences gives a degenerate interval that excludes zero
# whenever the differences share a sign (seen live 2026-11-25: n=2 "significant").
# McNemar's exact test handles small n correctly and has no such floor.
MIN_BOOTSTRAP_N = 10


def bootstrap_ci(
    baseline_values: list[float],
    candidate_values: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, bool]:
    """Paired bootstrap CI on the candidate-minus-baseline mean delta.
    Returns `(ci_low, ci_high, significant)`; significant means the CI
    excludes zero and there are at least `MIN_BOOTSTRAP_N` pairs."""
    n = len(baseline_values)
    if n == 0 or n != len(candidate_values):
        raise ValueError("baseline_values and candidate_values must be the same, nonzero length")

    diffs = [c - b for b, c in zip(baseline_values, candidate_values)]
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(sample) / n)
    resampled_means.sort()

    alpha = 1 - ci
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    ci_low = resampled_means[max(0, lo_idx)]
    ci_high = resampled_means[min(n_resamples - 1, hi_idx)]
    significant = (ci_low > 0 or ci_high < 0) and n >= MIN_BOOTSTRAP_N
    return ci_low, ci_high, significant


def mcnemar_exact(baseline_binary: list[bool], candidate_binary: list[bool]) -> tuple[float, bool]:
    """Exact (binomial) McNemar's test on paired binary outcomes. Returns
    `(p_value, significant)` at alpha=0.05. Only the discordant pairs
    (baseline != candidate) carry information; ties are ignored, matching
    McNemar's own definition."""
    from scipy.stats import binomtest

    if len(baseline_binary) != len(candidate_binary):
        raise ValueError("baseline_binary and candidate_binary must be the same length")

    b_to_f = sum(1 for b, c in zip(baseline_binary, candidate_binary) if b and not c)
    f_to_b = sum(1 for b, c in zip(baseline_binary, candidate_binary) if not b and c)
    discordant = b_to_f + f_to_b
    if discordant == 0:
        return 1.0, False

    p_value = float(binomtest(min(b_to_f, f_to_b), discordant, p=0.5).pvalue)
    return p_value, bool(p_value < 0.05)


def _is_scored(value) -> bool:
    """A metric value that can be averaged: present, and not NaN."""
    return value is not None and not math.isnan(float(value))


def _metric_deltas(
    metric_paths: tuple[str, ...],
    baseline_results: list[Result],
    candidate_results: list[Result],
) -> list[MetricDelta]:
    deltas = []
    for metric_name in metric_paths:
        pairs = [
            (_get_metric(b, metric_name), _get_metric(c, metric_name))
            for b, c in zip(baseline_results, candidate_results)
        ]
        # A pair is usable only when BOTH runs scored the case. NaN counts as
        # unscored, not as a value: RAGAS returns NaN for a case it can't score
        # (faithfulness of "I don't know" has no claims to check), and one NaN
        # in either run turns the whole paired comparison into nan — which the
        # 2026-09-22 HotpotQA diff reported as `nan, significant=False`,
        # silently voiding the one metric the regression was clearest on.
        # Same reading as `sagwa.gate.aggregate_metric`.
        pairs = [(b, c) for b, c in pairs if _is_scored(b) and _is_scored(c)]
        if not pairs:
            continue

        if metric_name in _BINARY_METRIC_PATHS:
            baseline_binary = [bool(b) for b, _ in pairs]
            candidate_binary = [bool(c) for _, c in pairs]
            baseline_mean = sum(baseline_binary) / len(pairs)
            candidate_mean = sum(candidate_binary) / len(pairs)
            p_value, significant = mcnemar_exact(baseline_binary, candidate_binary)
            deltas.append(
                MetricDelta(
                    metric_name=metric_name,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    delta=candidate_mean - baseline_mean,
                    test="mcnemar",
                    p_value=p_value,
                    significant=significant,
                    n=len(pairs),
                )
            )
        else:
            baseline_values = [float(b) for b, _ in pairs]
            candidate_values = [float(c) for _, c in pairs]
            baseline_mean = sum(baseline_values) / len(pairs)
            candidate_mean = sum(candidate_values) / len(pairs)
            ci_low, ci_high, significant = bootstrap_ci(baseline_values, candidate_values)
            deltas.append(
                MetricDelta(
                    metric_name=metric_name,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    delta=candidate_mean - baseline_mean,
                    test="bootstrap_ci",
                    ci_low=ci_low,
                    ci_high=ci_high,
                    significant=significant,
                    n=len(pairs),
                )
            )
    return deltas


def _run_level_deltas(baseline_results: list[Result], candidate_results: list[Result]) -> list[MetricDelta]:
    """Run-level metrics (e.g. tokens per successful task) are a ratio of run
    totals: there are no paired per-case values, so no significance test.
    They are reported as a plain delta marked `run_level`, never dropped and
    never called significant."""
    deltas = []
    for metric_name in run_level_paths():
        aggregator = METRICS[metric_name].aggregator
        baseline_value, candidate_value = aggregator(baseline_results), aggregator(candidate_results)
        if baseline_value is None and candidate_value is None:
            continue
        deltas.append(MetricDelta(
            metric_name=metric_name, baseline_mean=baseline_value, candidate_mean=candidate_value,
            delta=None if baseline_value is None or candidate_value is None else candidate_value - baseline_value,
            test="run_level", n=len(baseline_results),
        ))
    return deltas


def _most_affected(flips: list[CaseFlip], tag_sizes: dict[str, int]) -> list[str]:
    """Tag(s) where the largest share of cases flipped pass->fail, needing at
    least two flips. A share, not a count: a tag every case carries ("hard" on
    HotpotQA) would otherwise always win and say nothing."""
    counts: dict[str, int] = {}
    for flip in flips:
        if flip.direction == "pass_to_fail":
            for tag in flip.tags:
                counts[tag] = counts.get(tag, 0) + 1
    shares = {tag: n / tag_sizes[tag] for tag, n in counts.items() if n >= 2 and tag_sizes.get(tag)}
    top = max(shares.values(), default=0)
    return sorted(tag for tag, share in shares.items() if share == top) if top else []


def diff_runs(
    session,
    baseline_run_id: str,
    candidate_run_id: str,
    gates_config: dict | None = None,
    golden_cases: dict | None = None,
) -> DiffResult:
    """Entry point (FR-16). `gates_config` supplies the pass/fail definition
    (see `case_passes`) for FR-17's flip list; `golden_cases` (`case_id ->
    GoldenCase`) supplies `tags` for `by_tag` grouping — when omitted, tags
    are loaded from the dataset each run's `Result.run.dataset_path` points
    at (both runs must reference the same dataset for tags to line up)."""
    from sagwa.storage import Run

    baseline_by_case = load_run_results(session, baseline_run_id)
    candidate_by_case = load_run_results(session, candidate_run_id)

    shared_ids = sorted(set(baseline_by_case) & set(candidate_by_case))
    baseline_only = sorted(set(baseline_by_case) - set(candidate_by_case))
    candidate_only = sorted(set(candidate_by_case) - set(baseline_by_case))

    baseline_results = [baseline_by_case[cid] for cid in shared_ids]
    candidate_results = [candidate_by_case[cid] for cid in shared_ids]

    overall = _metric_deltas(_CONTINUOUS_METRIC_PATHS + _BINARY_METRIC_PATHS, baseline_results, candidate_results)
    overall += _run_level_deltas(baseline_results, candidate_results)

    if golden_cases is None:
        golden_cases = {}
        candidate_run = session.get(Run, candidate_run_id)
        if candidate_run is not None:
            from sagwa.datasets import load_golden_set

            for case in load_golden_set(candidate_run.dataset_path):
                golden_cases[case.id] = case

    by_tag: dict[str, list[str]] = {}
    for cid in shared_ids:
        case = golden_cases.get(cid)
        for tag in (case.tags if case else []):
            by_tag.setdefault(tag, []).append(cid)

    by_tag_deltas = {}
    for tag, tag_case_ids in by_tag.items():
        tag_baseline = [baseline_by_case[cid] for cid in tag_case_ids]
        tag_candidate = [candidate_by_case[cid] for cid in tag_case_ids]
        by_tag_deltas[tag] = _metric_deltas(
            _CONTINUOUS_METRIC_PATHS + _BINARY_METRIC_PATHS, tag_baseline, tag_candidate
        ) + _run_level_deltas(tag_baseline, tag_candidate)

    gates_config = gates_config or {}
    flips = []
    for cid in shared_ids:
        baseline_pass = case_passes(baseline_by_case[cid], gates_config)
        candidate_pass = case_passes(candidate_by_case[cid], gates_config)
        if baseline_pass is None or candidate_pass is None or baseline_pass == candidate_pass:
            continue
        direction = "pass_to_fail" if baseline_pass and not candidate_pass else "fail_to_pass"
        case = golden_cases.get(cid)
        flips.append(CaseFlip(case_id=cid, direction=direction, tags=case.tags if case else []))

    return DiffResult(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        overall=overall,
        by_tag=by_tag_deltas,
        flips=flips,
        baseline_only_case_ids=baseline_only,
        candidate_only_case_ids=candidate_only,
        most_affected_tags=_most_affected(flips, {tag: len(ids) for tag, ids in by_tag.items()}),
    )


def format_table(result: DiffResult) -> str:
    """Fixed-width CLI table (FR-19) — no `rich`/`tabulate` dependency,
    matching this project's hand-roll-over-add-a-dependency convention."""
    lines = [
        f"Diff: baseline={result.baseline_run_id} candidate={result.candidate_run_id}",
        "",
        f"{'metric':<30} {'baseline':>10} {'candidate':>10} {'delta':>10} {'effect':>9} {'test':>12} {'sig':>5} {'n':>5}",
    ]
    for m in result.overall:
        sig = "n/a" if m.test == "run_level" else str(m.significant)
        lines.append(
            f"{m.metric_name:<30} {_fmt(m.baseline_mean):>10} {_fmt(m.candidate_mean):>10} "
            f"{_fmt(m.delta):>10} {_effect(m.metric_name, m.delta):>9} {m.test:>12} {sig:>5} {m.n:>5}"
        )

    if result.most_affected_tags:
        tag = result.most_affected_tags[0]
        n = sum(1 for f in result.flips if f.direction == "pass_to_fail" and tag in f.tags)
        lines.append("")
        lines.append(f"Most affected: {', '.join(result.most_affected_tags)} "
                     f"({n} of its cases flipped pass→fail)")

    if result.flips:
        lines.append("")
        lines.append(f"Flips ({len(result.flips)}):")
        for flip in result.flips:
            lines.append(f"  {flip.case_id}: {flip.direction}")

    if result.baseline_only_case_ids:
        lines.append("")
        lines.append(f"Baseline-only cases: {', '.join(result.baseline_only_case_ids)}")
    if result.candidate_only_case_ids:
        lines.append("")
        lines.append(f"Candidate-only cases: {', '.join(result.candidate_only_case_ids)}")

    return "\n".join(lines)


def _effect(metric_name: str, delta: float | None) -> str:
    """`↑ better` / `↓ worse`: the arrow is the raw direction, the word comes
    from the registry's `higher_is_better`, so a cost increase reads as worse
    and a recovery-rate increase as better. Blank for an unregistered metric."""
    spec = METRICS.get(metric_name)
    if spec is None or delta is None:
        return ""
    if abs(delta) < 1e-12:
        return "same"
    better = (delta > 0) == spec.higher_is_better
    return f"{'↑' if delta > 0 else '↓'} {'better' if better else 'worse'}"


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"
