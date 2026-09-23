"""Fake target pipelines with *known* behaviour, so Sagwa's verdicts can be
checked against ground truth (see validate.py).

- GoodAdapter:      answers every golden case correctly (the baseline).
- RegressedAdapter: GoodAdapter with deliberate faults in the cases listed
                    in INJECTED_FAULTS; every other case is untouched.
- CosmeticAdapter:  GoodAdapter with harmless formatting changes only —
                    a fair evaluator should NOT call this a regression.
"""
import json
from pathlib import Path

from sagwa.adapters.base import AdapterResult

DATASET = Path(__file__).resolve().parents[2] / "golden_sets" / "demo_synthetic.jsonl"

# case_id -> (fault type, broken answer)
INJECTED_FAULTS = {
    "demo-001": ("wrong_fact", "Kyoto"),
    "demo-004": ("wrong_fact", "1939"),
    "demo-006": ("wrong_fact", "Venus"),
    "demo-013": ("pii_leak", "The city council approved funding for a new downtown library branch due to resident demand. Contact press@city.gov for details."),
    "demo-015": ("pii_leak", "TechCorp's quarterly revenue rose 15% to $2.3 billion on strong cloud demand. Call 555-123-4567 for investor info."),
    "demo-014": ("truncation", "A study found"),
    "demo-016": ("truncation", "Heavy rain and"),
    "demo-019": ("truncation", "Scientists discovered a"),
    "demo-022": ("label_flip", "negative"),
    "demo-023": ("label_flip", "positive"),
    "demo-028": ("label_flip", "finance"),
}


def _correct_answers() -> dict[str, str]:
    answers = {}
    for line in DATASET.read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            answers[case["input"]] = case.get("expected_output") or ", ".join(case["expected_labels"])
    return answers


def _case_ids() -> dict[str, str]:
    return {json.loads(l)["input"]: json.loads(l)["id"] for l in DATASET.read_text().splitlines() if l.strip()}


def _result(answer: str) -> AdapterResult:
    return AdapterResult(answer=answer, context=None, latency_ms=1, tokens=len(answer.split()), cost_usd=0.0)


class GoodAdapter:
    name = "good"

    def __init__(self):
        self.answers = _correct_answers()
        self.ids = _case_ids()

    def run(self, case_input: str) -> AdapterResult:
        return _result(self.answers[case_input])


class RegressedAdapter(GoodAdapter):
    name = "regressed"

    def run(self, case_input: str) -> AdapterResult:
        fault = INJECTED_FAULTS.get(self.ids[case_input])
        return _result(fault[1] if fault else self.answers[case_input])


class CosmeticAdapter(GoodAdapter):
    name = "cosmetic"

    def run(self, case_input: str) -> AdapterResult:
        answer = self.answers[case_input]
        if not answer.endswith("."):
            answer += "."
        return _result(answer.upper())
