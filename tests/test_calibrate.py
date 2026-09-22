"""The calibration harness's own logic (`benchmarks/calibrate.py`).

The number a calibration study reports is only meaningful if the prompt was
not tuned against the cases it is scored on — the 2026-09-20 study's threshold
sweep was, which is why it is labelled "not a result" in the report. These
tests pin the split that fixes it, plus the label binarization and the
per-case cache that lets an interrupted study resume.
"""
import json

import pytest

from benchmarks.calibrate import _judge_label, _split


def _labels(n_pass: int, n_fail: int) -> list[dict]:
    rows = [{"id": f"pass-{i}", "human_label": "pass", "helpfulness": 4} for i in range(n_pass)]
    rows += [{"id": f"fail-{i}", "human_label": "fail", "helpfulness": 0} for i in range(n_fail)]
    return rows


def test_split_halves_each_label_so_both_sides_stay_balanced():
    tune, holdout = _split(_labels(100, 100))

    assert len(tune) == len(holdout) == 100
    for half in (tune, holdout):
        assert sum(r["human_label"] == "pass" for r in half) == 50
        assert sum(r["human_label"] == "fail" for r in half) == 50


def test_split_is_disjoint_and_covers_everything():
    rows = _labels(100, 100)
    tune, holdout = _split(rows)

    tune_ids = {r["id"] for r in tune}
    holdout_ids = {r["id"] for r in holdout}
    assert tune_ids.isdisjoint(holdout_ids), "a case used for tuning must not also be reported on"
    assert tune_ids | holdout_ids == {r["id"] for r in rows}


def test_split_is_deterministic():
    """Re-running the study must not reshuffle which cases were held out —
    otherwise a prompt can be tuned against yesterday's holdout."""
    rows = _labels(50, 50)
    first = [r["id"] for r in _split(rows)[1]]
    second = [r["id"] for r in _split(rows)[1]]
    assert first == second


def test_split_handles_an_odd_number_of_cases():
    tune, holdout = _split(_labels(7, 5))
    assert len(tune) + len(holdout) == 12


@pytest.mark.parametrize(
    "score, expected",
    [(1.0, "pass"), (0.6, "pass"), (0.59, "fail"), (0.0, "fail"), (None, "no_score")],
)
def test_judge_label_binarizes_at_the_gate_threshold(score, expected):
    """0.6 is the same threshold config/gates.yaml gates judge.score on, so
    kappa measures agreement with the decision the gate actually makes."""
    assert _judge_label(score) == expected


def test_unparseable_score_counts_as_disagreement_not_a_dropped_case():
    """Dropping them would quietly shrink n and flatter the result."""
    assert _judge_label(None) == "no_score"


def test_score_all_caches_per_case_so_an_interrupted_study_resumes(monkeypatch, tmp_path):
    import benchmarks.calibrate as calibrate

    cache = tmp_path / "scores.jsonl"
    monkeypatch.setattr(calibrate, "_scores_path", lambda version: cache)
    monkeypatch.setattr(calibrate, "groq_llm_call", lambda *a, **k: (lambda prompt: "0.9"))

    calls = []

    def _score(llm_call, query, answer, **kwargs):
        calls.append(query)
        return 0.9, "0.9"

    monkeypatch.setattr(calibrate, "score_absolute_with_rationale", _score)

    rows = [{"id": "a", "prompt": "q1", "response": "r1", "human_label": "pass"},
            {"id": "b", "prompt": "q2", "response": "r2", "human_label": "fail"}]

    first = calibrate._score_all(rows, "v-test")
    assert first == {"a": 0.9, "b": 0.9}
    assert len(calls) == 2
    assert len(cache.read_text().splitlines()) == 2

    # Second pass: everything is cached, so no judge call is made at all.
    second = calibrate._score_all(rows, "v-test")
    assert second == first
    assert len(calls) == 2, "a cached case must not be re-scored"


def test_score_all_resumes_a_partial_cache(monkeypatch, tmp_path):
    import benchmarks.calibrate as calibrate

    cache = tmp_path / "scores.jsonl"
    cache.write_text(json.dumps({"id": "a", "judge_score": 0.4, "rationale": "0.4"}) + "\n")
    monkeypatch.setattr(calibrate, "_scores_path", lambda version: cache)
    monkeypatch.setattr(calibrate, "groq_llm_call", lambda *a, **k: (lambda prompt: "0.9"))

    scored = []

    def _score(llm_call, query, answer, **kwargs):
        scored.append(query)
        return 0.9, "0.9"

    monkeypatch.setattr(calibrate, "score_absolute_with_rationale", _score)

    rows = [{"id": "a", "prompt": "q1", "response": "r1", "human_label": "pass"},
            {"id": "b", "prompt": "q2", "response": "r2", "human_label": "fail"}]

    result = calibrate._score_all(rows, "v-test")

    assert scored == ["q2"], "only the case missing from the cache should be scored"
    assert result == {"a": 0.4, "b": 0.9}
