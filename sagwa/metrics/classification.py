"""Classification metrics (PRD FR-1, FR-8): scored against a golden case's
`expected_labels` — the list-of-labels ground truth `GoldenCase` has always
declared, but which nothing previously read (`compute_metrics()` only
branched on `expected_output`, a single reference string).

The adapter's `answer` is still just a string — this module doesn't
require adapters to return a structured label list. It parses `answer`
into a predicted label *set* by splitting on commas/semicolons/newlines,
so both a single-label adapter (`"positive"`) and a multi-label one
(`"billing, refund"`) work through the same code path. Precision/recall/F1
are computed set-wise per case (not scikit-learn's aggregate multi-label
metrics, which need the whole dataset at once) — right-sized for scoring
one case's prediction against one case's ground truth, matching this
project's "hand-roll the simple case, no extra dependency" convention
(see `reference.py::rouge_l_f1`).
"""
from __future__ import annotations

import re


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def parse_labels(text: str) -> set[str]:
    """Splits `text` on commas/semicolons/newlines into a normalized label
    set. A plain single-label answer (`"positive"`) parses to a one-element
    set, so single- and multi-label predictions share this same path."""
    parts = re.split(r"[,;\n]", text)
    return {_normalize_label(p) for p in parts if p.strip()}


def compute_classification_metrics(answer: str, expected_labels: list[str]) -> dict:
    """Set-based precision/recall/F1 and exact-set-match between the
    predicted labels (parsed from `answer`) and `expected_labels`
    (PRD FR-8's "ground truth exists" case, for classification tasks)."""
    predicted = parse_labels(answer)
    expected = {_normalize_label(label) for label in expected_labels}

    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "exact_set_match": 1.0 if predicted == expected else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
