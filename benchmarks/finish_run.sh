#!/usr/bin/env bash
# Finishes the benchmark evidence: HotpotQA v2, then diff + cluster against v1.
#
# Everything else is already done and is NOT re-run:
#   - HotpotQA v1   : in benchmarks.db (the baseline)
#   - CNN/DailyMail : in benchmarks.db, gate passed
#   - Calibration   : calibration/calibration_*.json (kappa 0.450)
#
# v2 MUST run on the same model as v1 (openai/gpt-oss-120b) — a different
# target model would conflate the model change with the retrieval/prompt
# change under test, which is the whole thing being measured.
#
# Budget: ~150K tokens on gpt-oss-120b, whose free tier allows 200K/day.
# Run this when that day's budget is untouched, or on a paid tier.
#
#   ./benchmarks/finish_run.sh              # all 50 cases
#   CASES=25 ./benchmarks/finish_run.sh     # half, if the budget is tight
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export DATABASE_URL="sqlite:///./benchmarks/benchmarks.db"
export SAGWA_RAGAS_MODEL="openai/gpt-oss-120b"   # same as v1
export SAGWA_RAGAS_METRICS="faithfulness"        # same as v1
export PYTHONUNBUFFERED=1
OUT=benchmarks/out; mkdir -p "$OUT"
G=benchmarks/gates_hotpotqa.yaml
DATASET=golden_sets/hotpotqa.jsonl

V1=$(sqlite3 benchmarks/benchmarks.db "select id from runs where model like '%top_k=3%' order by rowid limit 1")
[ -n "$V1" ] || { echo "FATAL: no v1 baseline run in benchmarks.db — nothing to diff against."; exit 1; }
echo "baseline v1 = $V1"

if [ -n "${CASES:-}" ]; then
  DATASET=$(mktemp /tmp/hotpotqa_subset_XXXX.jsonl)
  head -n "$CASES" golden_sets/hotpotqa.jsonl > "$DATASET"
  echo "running a $CASES-case subset ($DATASET)"
fi

echo "== sagwa run hotpotqa v2"
V2=$(sagwa run --target benchmarks.adapters:HotpotV2 --dataset "$DATASET" --concurrency 2 \
     | tee /dev/stderr | sed -n 's/^Run \([^:]*\):.*/\1/p')

# A run that lost cases to rate limits cannot support a paired comparison —
# say so loudly rather than printing a confident diff over the survivors.
FAILED=$(sqlite3 benchmarks/benchmarks.db "select count(*) from results where run_id='$V2' and error is not null")
TOTAL=$(sqlite3 benchmarks/benchmarks.db "select count(*) from results where run_id='$V2'")
echo "v2 = $V2 ($FAILED of $TOTAL cases errored)"
if [ "$FAILED" -gt $((TOTAL / 10)) ]; then
  echo
  echo "!! ABORTING the diff: $FAILED of $TOTAL cases failed (>10%)."
  echo "!! Check the error with:"
  echo "   sqlite3 benchmarks/benchmarks.db \"select distinct substr(error,1,200) from results where run_id='$V2' and error is not null\""
  echo "!! If it is 'tokens per day (TPD)', the daily budget is gone — rerun on a fresh day or a paid tier."
  exit 1
fi

echo "hotpotqa_v1=$V1 hotpotqa_v2=$V2" > "$OUT/run_ids.txt"
echo; echo "== gate hotpotqa v2"; sagwa gate --run-id "$V2" --config $G --json "$OUT/gate_hotpotqa_v2.json" || true
echo; echo "== diff v1 -> v2"; sagwa diff --baseline "$V1" --candidate "$V2" --gates-config $G --json "$OUT/diff_hotpotqa.json"
echo; echo "== cluster v2 failures"; sagwa cluster --run-id "$V2" --gates-config $G --json "$OUT/clusters_hotpotqa_v2.json" || true
echo; echo "== done. run ids: $(cat $OUT/run_ids.txt)"
