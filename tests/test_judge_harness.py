from sagwa.judge.harness import score_absolute, score_pairwise


def test_score_absolute_parses_number_in_range():
    assert score_absolute(lambda prompt: "0.8", query="q", answer="a") == 0.8


def test_score_absolute_returns_none_for_unparseable_response():
    assert score_absolute(lambda prompt: "definitely great", query="q", answer="a") is None


def test_score_pairwise_parses_a_b_and_tie():
    assert score_pairwise(lambda prompt: "A", query="q", answer_a="x", answer_b="y") == "A"
    assert score_pairwise(lambda prompt: "b is better", query="q", answer_a="x", answer_b="y") == "B"
    assert score_pairwise(lambda prompt: "tie", query="q", answer_a="x", answer_b="y") == "tie"


def test_score_pairwise_returns_none_for_unparseable_response():
    assert score_pairwise(lambda prompt: "unclear", query="q", answer_a="x", answer_b="y") is None
