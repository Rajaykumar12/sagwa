# Judge Calibration Report

**Verdict (updated 2026-11-19): no judge tested clears κ ≥ 0.70, and the naive one-line judge beats the rubric.** On 100 held-out human labels, with the same model (`gpt-oss-120b`) and the same cases, the naive judge scores **κ = 0.66** [0.50, 0.80] against the v2 rubric's **κ = 0.44** [0.26, 0.61]: **Δκ = +0.22, 95% CI [+0.10, +0.36]**. FR-15a's comparison came out the opposite way to the one it was designed to show. See §7 for the full iteration history. `require_calibration()` still refuses every judge measured here.

The sections below (§1–§6) are the original 2026-09-20 study of prompt v1 and are kept unchanged as the baseline record (FR-14).

| | |
|---|---|
| Judge prompt version | `v1-default-rubric` |
| Judge model | `openai/gpt-oss-20b`, `temperature=0.0`, `reasoning_effort="low"`, `max_tokens=512` |
| Human labels | 200 (HelpSteer2 validation, balanced 100 pass / 100 fail) |
| Pass threshold | judge score ≥ 0.6 — the same threshold `config/gates.yaml` gates `judge.score` on |
| Accuracy | 72.5% |
| **Cohen's κ** | **0.450** (95% bootstrap CI **[0.325, 0.569]**, 2000 resamples) |
| Unscoreable cases | **0 / 200** |
| Run date | 2026-09-20 |
| Artifact | `calibration_v1-default-rubric_2026-09-20T144450.406736+0000.json` |

Even the *top* of the confidence interval (0.569) sits below the 0.70 bar, so this isn't a sample-size accident.

## 1. The human-labeled set

HelpSteer2's validation split: real prompt/response pairs rated by human annotators for helpfulness on a 0–4 scale. Sampled deterministically by `benchmarks/build.py` (fixed seed, single-turn prompts only, ≤2000 characters), balanced 100 pass / 100 fail, where **pass = helpfulness ≥ 3**.

**The honest caveat, stated up front**: these are real human labels, but they rate *HelpSteer2's* responses, not the output of a Sagwa target pipeline. This calibrates the judge as a general-purpose answer-quality rater. A judge gating a specific pipeline should ultimately be calibrated against labels drawn from that pipeline's own outputs — this is the closest defensible substitute that doesn't require inventing labels.

## 2. Where the disagreement is

Confusion matrix (rows = human, columns = judge):

| | judge: pass | judge: fail |
|---|---|---|
| **human: pass** | 83 | 17 |
| **human: fail** | **38** | 62 |

The error is **one-sided: the judge is too generous.** 38 responses humans rated unhelpful were passed, against 17 helpful ones failed.

But the judge is not guessing. Its mean score rises monotonically with human helpfulness:

| Human helpfulness | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| n | 28 | 38 | 34 | 46 | 54 |
| Mean judge score | 0.20 | 0.35 | **0.60** | 0.74 | 0.80 |

**The ordering is right; the cut point is wrong.** Helpfulness-2 — "partially helpful", which this study's binarization calls a fail — averages exactly 0.60, sitting precisely on the pass threshold. That band is a coin flip by construction: **22 of the 38 false passes are helpfulness-2 cases**.

Excluding that ambiguous band entirely (n=34 dropped, 166 remaining) lifts κ to **0.586** — still short of 0.70, but it locates most of the damage in one place.

Two representative false passes, both scored exactly 0.6 against human helpfulness 2:

- A response opening "I'm not sure I fully understand what you're asking for…" then guessing at the intent. The judge rewarded the attempt; the human penalized the miss.
- A troubleshooting answer giving generic "check your settings" steps that never address the specific configuration asked about. Plausible, fluent, unhelpful.

Both are the same failure mode: **the judge rewards well-formed, on-topic text that does not actually help.**

## 3. Threshold sweep (and why it isn't a fix)

| Threshold | 0.50 | 0.60 | **0.65** | **0.70** | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|
| κ | 0.440 | 0.450 | **0.500** | **0.500** | 0.460 | 0.460 | 0.380 |

Moving the pass line to 0.65–0.70 buys κ = 0.50. **This number should not be reported as a result**: the threshold was chosen by looking at the same 200 cases it is scored on, so it is fitted to this sample. A defensible version needs a held-out split — tune on one half, report on the other. The sweep's real value is diagnostic: the curve is flat and low across every threshold, which says the ceiling is the *prompt*, not the cut point.

## 4. What to change before re-running (FR-13 iteration)

In rough order of expected value:

1. **Give the judge the ground truth it's being asked to approximate.** `_ABSOLUTE_PROMPT` currently sees only question + answer. It cannot check correctness, so it grades fluency and topicality. The same blind spot showed up on HotpotQA, where a wrong answer scored 1.0, and it is why a correct "I don't know" is penalized. Passing the expected answer (and, for RAG, the retrieved context) is the single biggest lever.
2. **Define the scale in the rubric.** `DEFAULT_RUBRIC` describes 0.0 and 1.0 and leaves everything between undefined — so the model's "0.6" has no agreed meaning, which is exactly where the 22 helpfulness-2 cases pile up. Anchor each band with a short description.
3. **Add few-shot examples** drawn from the labeled set — one clear pass, one clear fail, and critically one *borderline* case with its label.
4. **Re-run against a held-out split** so the reported κ isn't fitted to the tuning data.

Each change is a new `judge_prompt_version`, producing a new artifact rather than overwriting this one (FR-14). This report is version `v1-default-rubric`'s honest baseline.

## 5. FR-15a baseline comparison

Run on 2026-11-19. See §7.

## 6. What this result is worth

A κ of 0.45 is a *finding*, not a failure of the project. It says: an off-the-shelf LLM judge with a one-line rubric agrees with human raters well enough to rank outputs, and not well enough to block a merge. Shipping that judge as a quality gate — which is what "we use LLM-as-judge" usually means in practice — would fail roughly **two in five unhelpful answers straight through**.

Measuring that is the point of the calibration requirement. The alternative, quietly gating on an uncalibrated judge, is the failure mode Sagwa exists to prevent.

## 7. Iteration history (2026-11-19): prompt v2, a bigger model, and FR-15a

All numbers here are on the **held-out half**: 100 cases (50 pass / 50 fail), split deterministically and stratified by label (`benchmarks/calibrate.py::_split`). No prompt or threshold decision was made on them. v1's 0.450 above is on all 200 cases; on the held-out half alone it is 0.46.

| Judge | Model | κ (95% CI) | Accuracy | False passes | False fails |
|---|---|---|---|---|---|
| v1 rubric (2026-09-20) | `gpt-oss-20b` | 0.46 [0.29, 0.63] | 73% | 19 | 8 |
| v2 banded rubric + few-shot | `gpt-oss-20b` | **0.32** [0.15, 0.49] | 66% | 27 | 7 |
| v2 banded rubric + few-shot | `gpt-oss-120b` | 0.44 [0.26, 0.61] | 72% | 17 | 11 |
| **Naive: "Rate how good this answer is."** | `gpt-oss-120b` | **0.66** [0.50, 0.80] | **83%** | 10 | 7 |

CIs are 4,000-resample bootstraps over the 100 cases. Unscoreable cases: 0 for every judge. Artifacts: `calibration_v2-banded-rubric_*.json`, `calibration_v2-banded-rubric@gpt-oss-120b_*.json`, `calibration_v0-naive-baseline@gpt-oss-120b_naive_judge_*.json`. Per-case scores and rationales: `helpsteer2_judge_scores_<version>.jsonl`.

### What each step showed

1. **Prompt v2 made the 20b judge worse** (0.46 → 0.32), and more generous (27 false passes). It scored 0.54 on the tune half, so the tune/holdout gap is itself ±0.2 at n = 100. One half is not enough to iterate a prompt against with any precision.
2. **Choosing the pass threshold on the tune half doesn't help.** For v1 and v2 alike, the best tune-half threshold is 0.5, giving held-out κ of 0.44 and 0.32. The threshold was never the ceiling.
3. **A larger model helps the rubric only back to v1's level** (v2: 0.32 → 0.44 on 120b).
4. **FR-15a: the naive judge beats the v2 rubric by +0.22 on the same model**, CI [+0.10, +0.36], and no bootstrap resample reverses it. The naive prompt was fixed in `calibrate.py` before any of these runs, so it was not tuned to win.

### Why the rubric loses here

Mean judge score by human helpfulness (held-out half):

| Helpfulness | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| v2 rubric @ 120b | 0.28 | 0.34 | **0.60** | 0.74 | 0.76 |
| naive @ 120b | 0.24 | 0.25 | **0.44** | 0.71 | 0.76 |

The binarization calls helpfulness 2 ("partially helpful") a **fail**. The v2 rubric's band anchors describe a partially correct answer as about 0.6, which is exactly the pass line, so the bands encode a different definition of "pass" than the labels do. The naive judge has no anchors and places that band below the line. **A rubric is a claim about what each score means; when its claim disagrees with the labels' definition of pass, a more detailed rubric agrees less.**

### Caveats

- **v2's main feature went unused.** v2 was written to grade against the expected answer and retrieved context, which HelpSteer2 doesn't have. On a reference-bearing task (the HotpotQA regression, where v1 rated the worse version +0.22 higher), v2 may still beat the naive judge. That is unmeasured: there are no human labels on HotpotQA outputs.
- **The naive judge ran only on 120b.** Its 20b run hit the model's daily token cap at 109 of 200 cases. The comparison is valid because both judges share a model, but it is not the default judge model.
- **0.66 is not a pass.** The point estimate is below 0.70 and the CI spans it.

### Recommendation (not applied)

Switching the default judge is a product decision, and changing the prompt invalidates every stored `judge.score`. Recommended: make the naive prompt on `gpt-oss-120b` the default *general-quality* judge, keep v2 for reference-bearing tasks, and collect about 100 human labels on HotpotQA outputs to test v2 where it was designed to work. Until a judge clears 0.70 on held-out data, `judge.score` stays out of any gate by itself.
