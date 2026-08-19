from sagwa.datasets.schema import GoldenCase
from sagwa.metrics import compute_metrics
from sagwa.metrics.classification import compute_classification_metrics, parse_labels


def test_parse_labels_single_label():
    assert parse_labels("positive") == {"positive"}


def test_parse_labels_multi_label_comma_and_case_whitespace():
    assert parse_labels("Billing, Refund\nUrgent") == {"billing", "refund", "urgent"}


def test_parse_labels_ignores_empty_segments():
    assert parse_labels("positive, , ") == {"positive"}


def test_compute_classification_metrics_exact_single_label_match():
    result = compute_classification_metrics("positive", ["positive"])
    assert result == {"exact_set_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_compute_classification_metrics_single_label_mismatch():
    result = compute_classification_metrics("negative", ["positive"])
    assert result["exact_set_match"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_compute_classification_metrics_multi_label_partial_overlap():
    # predicted {billing, urgent}, expected {billing, refund} -> 1 true positive
    result = compute_classification_metrics("billing, urgent", ["billing", "refund"])
    assert result["exact_set_match"] == 0.0
    assert result["precision"] == 0.5  # 1 of 2 predicted labels correct
    assert result["recall"] == 0.5  # 1 of 2 expected labels found
    assert result["f1"] == 0.5


def test_compute_metrics_includes_classification_key_when_expected_labels_present():
    case = GoldenCase(
        id="c1", input="Classify sentiment", expected_labels=["positive"],
        task_type="classification", tags=[],
    )
    metrics = compute_metrics(case, answer="positive", context=None)
    assert "classification" in metrics
    assert metrics["classification"]["exact_set_match"] == 1.0
    assert "reference" not in metrics  # no expected_output on this case


def test_compute_metrics_omits_classification_key_without_expected_labels():
    case = GoldenCase(id="c2", input="Summarize this.", task_type="summarization", tags=[])
    metrics = compute_metrics(case, answer="A summary.", context=None)
    assert "classification" not in metrics
