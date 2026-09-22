from sagwa.judge.harness import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RUBRIC,
    groq_llm_call,
    score_absolute,
    score_absolute_with_rationale,
    score_pairwise,
)


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


def test_score_absolute_with_rationale_returns_score_and_raw_text():
    score, rationale = score_absolute_with_rationale(
        lambda prompt: "0.8 — mostly correct", query="q", answer="a"
    )
    assert score == 0.8
    assert rationale == "0.8 — mostly correct"


def test_score_absolute_with_rationale_returns_none_score_but_keeps_rationale():
    score, rationale = score_absolute_with_rationale(lambda prompt: "definitely great", query="q", answer="a")
    assert score is None
    assert rationale == "definitely great"


# --- prompt construction (judge v2) -------------------------------------
# The v1 judge saw only question + answer, scored kappa=0.450 against human
# labels, and rated a measurably worse pipeline 0.22 HIGHER than its baseline
# (docs/STATUS.md §2a). These tests pin the three things that fixed:
# ground truth reaches the judge, the scale's middle is defined, and the
# blocks are omitted rather than sent empty when a case has no reference.


def _capture_prompt(prompts: list):
    def _call(prompt: str) -> str:
        prompts.append(prompt)
        return "0.5"

    return _call


def test_prompt_includes_expected_answer_and_context_when_given():
    prompts = []
    score_absolute(
        _capture_prompt(prompts), query="q", answer="a", expected="the gold answer", context="retrieved text"
    )
    assert "Expected answer: the gold answer" in prompts[0]
    assert "retrieved text" in prompts[0]


def test_prompt_omits_reference_headings_when_case_has_none():
    """A HelpSteer2 helpfulness label has neither — the judge must not be
    shown an 'Expected answer:' heading with nothing under it."""
    prompts = []
    score_absolute(_capture_prompt(prompts), query="q", answer="a")
    assert "Expected answer:" not in prompts[0]
    assert "Context the answer had to rely on:" not in prompts[0]


def test_rubric_defines_the_middle_of_the_scale_not_just_the_endpoints():
    # 22 of 38 false passes in the 2026-09-20 study sat at exactly 0.6, the
    # band the v1 rubric never described.
    for band in ("0.2", "0.4", "0.6", "0.8"):
        assert f"{band} -" in DEFAULT_RUBRIC


def test_rubric_states_correctness_beats_fluency():
    assert "NOT fluency" in DEFAULT_RUBRIC


def test_few_shot_can_be_emptied_for_the_naive_baseline_judge():
    """FR-15a's baseline is this same harness with the rubric and examples
    stripped — same model, same parser, so the comparison isolates the prompt."""
    prompts = []
    score_absolute(_capture_prompt(prompts), query="q", answer="a", rubric="Rate it.", few_shot="")
    assert "Examples:" not in prompts[0]
    assert "Rate it." in prompts[0]


# --- model routing -------------------------------------------------------


class _FakeGroq:
    """Captures the model string `groq_llm_call` actually sends."""

    last_model = None

    def __init__(self, *args, **kwargs):
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, model, **kwargs):
        _FakeGroq.last_model = model

        class _Message:
            content = "0.7"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


def test_judge_model_defaults_when_no_env_var(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("SAGWA_JUDGE_MODEL", raising=False)
    monkeypatch.setattr("groq.Groq", _FakeGroq)
    groq_llm_call()("prompt")
    assert _FakeGroq.last_model == DEFAULT_JUDGE_MODEL


def test_judge_model_reads_sagwa_judge_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("SAGWA_JUDGE_MODEL", "some/other-model")
    monkeypatch.setattr("groq.Groq", _FakeGroq)
    groq_llm_call()("prompt")
    assert _FakeGroq.last_model == "some/other-model"


def test_explicit_model_argument_wins_over_env(monkeypatch):
    """`sagwa cluster` passes SAGWA_CLUSTER_MODEL explicitly — an explicit
    argument must not be overridden by the judge's own env var."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("SAGWA_JUDGE_MODEL", "judge/model")
    monkeypatch.setattr("groq.Groq", _FakeGroq)
    groq_llm_call("cluster/model")("prompt")
    assert _FakeGroq.last_model == "cluster/model"
