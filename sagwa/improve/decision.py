"""Accept or reject a candidate prompt revision (PRD FR-39).

Accepted only when, against the baseline run:
1. the primary metric (default `task.success`) changed significantly, in its
   registered good direction; and
2. no metric in the gate config got significantly worse.

Both judgments come from `sagwa diff` (paired bootstrap CI / exact McNemar's)
and the gate's own direction rule, so `improve` can never call something an
improvement that `sagwa diff` would call noise. A loop that always "improves"
is measuring noise; on a small golden set this rejects most candidates, and
that is the intended behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sagwa.diff import DiffResult
from sagwa.gate import _is_worse
from sagwa.metrics.registry import METRICS


@dataclass
class Decision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "reasons": self.reasons, "regressed": self.regressed}


def _describe(m) -> str:
    stat = f"p={m.p_value:.3f}" if m.p_value is not None else (
        f"95% CI [{m.ci_low:+.3f}, {m.ci_high:+.3f}]" if m.ci_low is not None else "no test")
    return f"{m.metric_name} {m.baseline_mean:.3f} → {m.candidate_mean:.3f} ({m.delta:+.3f}, {stat})"


def decide(diff: DiffResult, gates_config: dict, primary: str = "task.success", tolerance: float = 0.0) -> Decision:
    deltas = {m.metric_name: m for m in diff.overall}
    p = deltas.get(primary)
    if p is None or p.delta is None or p.baseline_mean is None:
        return Decision(False, [f"{primary} was not computed in both runs"])

    higher_is_better = METRICS[primary].higher_is_better if primary in METRICS else True
    right_way = (p.delta > 0) == higher_is_better
    if not (p.significant and right_way and abs(p.delta) > tolerance):
        why = ("no change" if abs(p.delta) < 1e-12 else "worse" if not right_way
               else f"within the {tolerance} tolerance" if abs(p.delta) <= tolerance else "not significant")
        return Decision(False, [f"{_describe(p)}: {why}"])

    regressed = [
        name for name, rule in gates_config.items()
        if name != primary and (m := deltas.get(name)) is not None and m.delta is not None
        and m.significant and _is_worse(name, m.delta, rule["op"], tolerance)
    ]
    if regressed:
        return Decision(False, [f"{_describe(p)}: improved, but a gated metric regressed: "
                                + "; ".join(_describe(deltas[n]) for n in regressed)], regressed)
    return Decision(True, [f"{_describe(p)}: significant improvement, no gated metric regressed"])
