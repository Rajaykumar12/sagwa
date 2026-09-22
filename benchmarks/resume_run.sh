#!/usr/bin/env bash
# Resume from v2, with calibration running in parallel on its own model pool.
#   v2       : gpt-oss-120b  (MUST match v1's model — it is the baseline)
#   cnndm    : qwen3.8-27b   (no baseline, so a free pool costs nothing)
#   RAGAS    : gpt-oss-120b, faithfulness only
#   judge    : gpt-oss-20b   (shared with calibration, which runs concurrently)
set -euo pipefail
cd "/run/media/rajay/New Volume/Machine Learning/Sagwa"
export PATH="$PWD/.venv/bin:$PATH"
export DATABASE_URL="sqlite:///./benchmarks/benchmarks.db"
export SAGWA_RAGAS_MODEL="openai/gpt-oss-120b"
export SAGWA_RAGAS_METRICS="faithfulness"
export PYTHONUNBUFFERED=1
OUT=benchmarks/out; mkdir -p "$OUT"
G=benchmarks/gates_hotpotqa.yaml
V1=$(sqlite3 benchmarks/benchmarks.db "select id from runs order by rowid limit 1")

# Calibration is independent of every run below and only touches the judge
# model, so it overlaps instead of queueing behind them. It is resumable.
echo "== judge calibration (HelpSteer2) started in parallel"
python -m benchmarks.calibrate > "$OUT/calibration.log" 2>&1 &
CALIB=$!

run() { sagwa run --target "benchmarks.adapters:$1" --dataset "$2" --concurrency 2 | tee /dev/stderr | sed -n 's/^Run \([^:]*\):.*/\1/p'; }

echo "== sagwa run hotpotqa v2"; V2=$(run HotpotV2 golden_sets/hotpotqa.jsonl)
echo "== sagwa run cnndm";       CN=$(SAGWA_TARGET_MODEL=qwen/qwen3.8-27b run CnnDmSummarizer golden_sets/cnndm.jsonl)
echo "hotpotqa_v1=$V1 hotpotqa_v2=$V2 cnndm=$CN" > "$OUT/run_ids.txt"

echo "== gate hotpotqa v1"; sagwa gate --run-id "$V1" --config $G --json "$OUT/gate_hotpotqa_v1.json" || true
echo "== gate hotpotqa v2"; sagwa gate --run-id "$V2" --config $G --json "$OUT/gate_hotpotqa_v2.json" || true
echo "== gate cnndm"; sagwa gate --run-id "$CN" --config benchmarks/gates_cnndm.yaml --json "$OUT/gate_cnndm.json" || true
echo "== diff hotpotqa v1 -> v2"; sagwa diff --baseline "$V1" --candidate "$V2" --gates-config $G --json "$OUT/diff_hotpotqa.json"
echo "== cluster hotpotqa v2"; sagwa cluster --run-id "$V2" --gates-config $G --json "$OUT/clusters_hotpotqa_v2.json"

echo "== waiting for calibration"; wait $CALIB && tail -20 "$OUT/calibration.log"
echo "== done: $(cat $OUT/run_ids.txt)"
