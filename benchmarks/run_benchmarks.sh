#!/usr/bin/env bash
# Sagwa end to end on public benchmarks, using only the real CLI.
#   HotpotQA: baseline HotpotV1 vs candidate HotpotV2 (the "PR under review")
#   CNN/DM:   one run of the summarization target (PRD FR-3c)
# Then the judge calibration against HelpSteer2 human labels.
# Needs GROQ_API_KEY in .env. Uses its own DB (benchmarks/benchmarks.db).
set -euo pipefail
cd "$(dirname "$0")/.."

export DATABASE_URL="sqlite:///./benchmarks/benchmarks.db"
# Groq's free tier rate-limits tokens per minute PER MODEL, so the three LLM
# roles run on three models: targets and RAGAS on gpt-oss-120b, the judge on
# gpt-oss-20b (calibration has to measure the same judge the gate uses).
# context_precision is skipped because no gate config here reads it.
export SAGWA_RAGAS_MODEL="openai/gpt-oss-120b"
export SAGWA_RAGAS_METRICS="faithfulness"
OUT=benchmarks/out
mkdir -p "$OUT"
rm -f benchmarks/benchmarks.db
alembic upgrade head >/dev/null 2>&1

run() {
  sagwa run --target "benchmarks.adapters:$1" --dataset "$2" --concurrency 2 \
    | tee /dev/stderr | sed -n 's/^Run \([^:]*\):.*/\1/p'
}

echo "== sagwa run hotpotqa v1"; V1=$(run HotpotV1 golden_sets/hotpotqa.jsonl)
echo "== sagwa run hotpotqa v2"; V2=$(run HotpotV2 golden_sets/hotpotqa.jsonl)
echo "== sagwa run cnndm";       CN=$(run CnnDmSummarizer golden_sets/cnndm.jsonl)
echo "hotpotqa_v1=$V1 hotpotqa_v2=$V2 cnndm=$CN" > "$OUT/run_ids.txt"

G=benchmarks/gates_hotpotqa.yaml
echo; echo "== gate hotpotqa v1"; sagwa gate --run-id "$V1" --config $G --json "$OUT/gate_hotpotqa_v1.json" || true
echo; echo "== gate hotpotqa v2"; sagwa gate --run-id "$V2" --config $G --json "$OUT/gate_hotpotqa_v2.json" || true
echo; echo "== gate cnndm"; sagwa gate --run-id "$CN" --config benchmarks/gates_cnndm.yaml --json "$OUT/gate_cnndm.json" || true
echo; echo "== diff hotpotqa v1 -> v2"; sagwa diff --baseline "$V1" --candidate "$V2" --gates-config $G --json "$OUT/diff_hotpotqa.json"
echo; echo "== cluster hotpotqa v2"; sagwa cluster --run-id "$V2" --gates-config $G --json "$OUT/clusters_hotpotqa_v2.json"

echo; echo "== judge calibration (HelpSteer2)"; python -m benchmarks.calibrate

echo; echo "Run ids: $(cat $OUT/run_ids.txt)"
echo "Browse it: DATABASE_URL=$DATABASE_URL sagwa dashboard"
