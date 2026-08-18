from sagwa.datasets.schema import GoldenCase
from sagwa.metrics import compute_metrics
from sagwa.metrics.reference import compute_reference_metrics, exact_match, fuzzy_match, rouge_l_f1
from sagwa.metrics.safety import detect_pii, detect_toxicity


def test_exact_match():
    assert exact_match("Paris", "  paris ") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_fuzzy_match_is_between_zero_and_one():
    assert fuzzy_match("The cat sat on the mat", "The cat sat on the mat") == 1.0
    score = fuzzy_match("The cat sat on the mat", "A dog ran in the park")
    assert 0.0 <= score < 1.0


def test_rouge_l_f1_perfect_and_partial_overlap():
    assert rouge_l_f1("the quick brown fox", "the quick brown fox") == 1.0
    partial = rouge_l_f1("the quick brown fox", "the slow brown fox jumps")
    assert 0.0 < partial < 1.0


def test_rouge_l_f1_handles_empty_strings():
    assert rouge_l_f1("", "something") == 0.0
    assert rouge_l_f1("something", "") == 0.0


def test_compute_reference_metrics_shape():
    metrics = compute_reference_metrics("Paris is the capital of France.", "Paris is the capital of France")
    assert set(metrics) >= {"exact_match", "fuzzy_match", "rouge_l_f1"}
    assert metrics["exact_match"] in (0.0, 1.0)


def test_detect_pii_flags_email_and_phone():
    result = detect_pii("Contact me at jane.doe@example.com or 415-555-0100.")
    assert result["flagged"] is True
    assert "email" in result["categories"]


def test_detect_pii_clean_text():
    result = detect_pii("The answer is 42.")
    assert result["flagged"] is False
    assert result["categories"] == []


def test_detect_toxicity_flags_seed_keyword():
    assert detect_toxicity("You're an idiot.")["flagged"] is True
    assert detect_toxicity("This is a helpful answer.")["flagged"] is False


def test_compute_metrics_includes_safety_always_and_reference_when_expected_output_present():
    case_with_reference = GoldenCase(
        id="c1", input="What is the capital of France?", expected_output="Paris",
        task_type="rag_qa", tags=[],
    )
    metrics = compute_metrics(case_with_reference, answer="Paris", context=None)
    assert "safety" in metrics
    assert "reference" in metrics
    assert "ragas" not in metrics  # no context supplied, so RAGAS metrics don't apply

    case_without_reference = GoldenCase(id="c2", input="Summarize this.", task_type="summarization", tags=[])
    metrics_no_ref = compute_metrics(case_without_reference, answer="A summary.", context=None)
    assert "reference" not in metrics_no_ref
