"""Calibrates Sagwa's judge against HelpSteer2's human helpfulness ratings
(PRD FR-13..FR-15a), using the labels built by `benchmarks.build`.

    python -m benchmarks.calibrate                # tune half + held-out half
    python -m benchmarks.calibrate --baseline     # also score the FR-15a naive judge

**The split is the point.** The 2026-09-20 study reported kappa on the same 200
cases a threshold was chosen from, which flatters the result. Cases are split
50/50 within each human label (stratified, deterministic): iterate the prompt
against `tune`, and report the number from `holdout`, which no decision was
made on.

FR-15a: `--baseline` scores a naive one-line judge — no rubric, no bands, no
examples — through the same model, temperature and parser. Only the prompt
differs, so the reported delta isolates the rubric rather than model choice
(docs/adr/0005-public-benchmarks-over-ringo.md).

Per-case scores are cached per judge version, so an interrupted run resumes
and a re-run of an unchanged judge costs nothing.

Caveat that belongs in every report built from this: real human labels, but on
HelpSteer2's own responses, not on a Sagwa target pipeline's output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from sagwa.judge.calibration import calibrate, save_calibration_result
from sagwa.judge.harness import JUDGE_PROMPT_VERSION, groq_llm_call, score_absolute_with_rationale

PASS_THRESHOLD = 0.6  # matches config/gates.yaml's judge.score gate
LABELS = Path("calibration/helpsteer2_human_labels.jsonl")

NAIVE_PROMPT_VERSION = "v0-naive-baseline"
NAIVE_RUBRIC = "Rate how good this answer is."
NAIVE_FEW_SHOT = ""


def _scores_path(prompt_version: str) -> Path:
    return Path(f"calibration/helpsteer2_judge_scores_{prompt_version}.jsonl")


def _score_all(rows: list[dict], prompt_version: str, **judge_kwargs) -> dict[str, float | None]:
    """Scores every row, caching per case so an interrupted run resumes."""
    path = _scores_path(prompt_version)
    done: dict[str, dict] = {}
    if path.exists():
        done = {s["id"]: s for s in map(json.loads, path.read_text().splitlines())}

    llm_call = groq_llm_call()
    with path.open("a") as out:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            score, rationale = score_absolute_with_rationale(
                llm_call, query=row["prompt"], answer=row["response"], **judge_kwargs
            )
            done[row["id"]] = {"id": row["id"], "judge_score": score, "rationale": rationale}
            out.write(json.dumps(done[row["id"]]) + "\n")
            out.flush()
            print(f"[{prompt_version}] [{i + 1}/{len(rows)}] {row['id']} "
                  f"human={row['human_label']} judge={score}", flush=True)
    return {case_id: s["judge_score"] for case_id, s in done.items()}


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified 50/50 by human label, deterministic: alternate within each
    label so both halves keep the 100-pass/100-fail balance the labels were
    built with."""
    tune, holdout = [], []
    for label in ("pass", "fail"):
        in_label = [r for r in rows if r["human_label"] == label]
        tune += in_label[0::2]
        holdout += in_label[1::2]
    return tune, holdout


def _judge_label(score: float | None) -> str:
    # An unparseable score is a disagreement, not a dropped case: counting it
    # as "no_score" keeps n honest rather than quietly shrinking the set.
    if score is None:
        return "no_score"
    return "pass" if score >= PASS_THRESHOLD else "fail"


def _report(rows: list[dict], scores: dict[str, float | None], version: str, baseline_name=None):
    human = [r["human_label"] for r in rows]
    judge = [_judge_label(scores[r["id"]]) for r in rows]
    result = calibrate(human, judge, judge_prompt_version=version, baseline_name=baseline_name)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"saved -> {save_calibration_result(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="also score the FR-15a naive judge on the same held-out cases")
    args = parser.parse_args()
    load_dotenv()

    rows = [json.loads(line) for line in LABELS.read_text().splitlines()]
    tune, holdout = _split(rows)
    print(f"{len(rows)} labels -> tune={len(tune)} holdout={len(holdout)}")

    scores = _score_all(rows, JUDGE_PROMPT_VERSION)
    print("\n=== TUNE half (iterate the prompt against this; NOT the reported number)")
    _report(tune, scores, f"{JUDGE_PROMPT_VERSION}-tune")
    print("\n=== HOLDOUT half (the honest number — no decisions were made on these cases)")
    _report(holdout, scores, JUDGE_PROMPT_VERSION)

    if args.baseline:
        print("\n=== FR-15a baseline: naive one-line judge, same model/temperature")
        naive = _score_all(rows, NAIVE_PROMPT_VERSION, rubric=NAIVE_RUBRIC, few_shot=NAIVE_FEW_SHOT)
        _report(holdout, naive, NAIVE_PROMPT_VERSION, baseline_name="naive_judge")


if __name__ == "__main__":
    main()
