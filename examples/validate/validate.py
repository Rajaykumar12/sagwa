"""Does Sagwa actually catch regressions? A controlled experiment.

Runs the demo golden set through three fake pipelines (examples/validate/
adapters.py) whose faults are known in advance, drives the real `sagwa`
CLI commands (run -> gate -> diff -> cluster), and scores Sagwa's verdicts
against that ground truth.

    python examples/validate/validate.py           # offline, deterministic
    python examples/validate/validate.py --judge   # also call the Groq judge

Uses its own throwaway DB (examples/validate/validate.db), never sagwa.db.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = HERE / "validate.db"
GATES = HERE / "gates.yaml"
DATASET = ROOT / "golden_sets" / "demo_synthetic.jsonl"

# Must be set before sagwa.storage is imported (it reads DATABASE_URL lazily,
# but alembic below needs it in the subprocess env too).
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"
sys.path.insert(0, str(ROOT))

from typer.testing import CliRunner  # noqa: E402

import sagwa.storage.db  # noqa: E402,F401  (triggers load_dotenv)
from sagwa.cli import app  # noqa: E402
from examples.validate.adapters import INJECTED_FAULTS  # noqa: E402

if "--judge" not in sys.argv:
    # load_dotenv() just pulled GROQ_API_KEY from .env; drop it so the run is
    # offline and deterministic (judge + cluster labels degrade gracefully).
    os.environ.pop("GROQ_API_KEY", None)

cli = CliRunner()
checks: list[tuple[str, bool, str]] = []


def sagwa(*args: str):
    result = cli.invoke(app, list(args))
    return result.exit_code, result.output


def sagwa_run(adapter: str) -> str:
    code, out = sagwa("run", "--target", f"examples.validate.adapters:{adapter}", "--dataset", str(DATASET))
    match = re.search(r"Run (\S+):", out)
    if code != 0 or not match:
        sys.exit(f"sagwa run failed for {adapter}:\n{out}")
    return match.group(1)


def sagwa_json(*args: str) -> tuple[int, dict | list]:
    out_path = HERE / "_tmp.json"
    code, out = sagwa(*args, "--json", str(out_path))
    data = json.loads(out_path.read_text())
    out_path.unlink()
    print(out)
    return code, data


def check(name: str, ok: bool, detail: str = ""):
    checks.append((name, ok, detail))


def main():
    DB.unlink(missing_ok=True)
    subprocess.run(["alembic", "upgrade", "head"], cwd=ROOT, check=True, capture_output=True)

    print("== Running three pipelines over", DATASET.name)
    good, bad, cosmetic = sagwa_run("GoodAdapter"), sagwa_run("RegressedAdapter"), sagwa_run("CosmeticAdapter")
    print(f"good={good}  regressed={bad}  cosmetic={cosmetic}\n")

    print("== H1/H2/H3: does the gate pass good and cosmetic, and fail regressed?")
    for label, run_id, should_pass in [("good", good, True), ("regressed", bad, False), ("cosmetic", cosmetic, True)]:
        code, _ = sagwa_json("gate", "--run-id", run_id, "--config", str(GATES))
        passed = code == 0
        check(f"gate {'passes' if should_pass else 'fails'} {label} run", passed == should_pass,
              f"exit code {code}")

    print("== H4: does diff flag exactly the injected cases (no more, no less)?")
    _, d = sagwa_json("diff", "--baseline", good, "--candidate", bad, "--gates-config", str(GATES))
    flagged = {f["case_id"] for f in d["flips"] if f["direction"] == "pass_to_fail"}
    injected = set(INJECTED_FAULTS)
    missed, false_alarms = injected - flagged, flagged - injected
    check("diff recall: every injected fault flagged", not missed, f"missed {sorted(missed)}" if missed else f"{len(injected)}/{len(injected)}")
    check("diff precision: no untouched case flagged", not false_alarms, f"false alarms {sorted(false_alarms)}" if false_alarms else "0 false alarms")

    sig = {m["metric_name"] for m in d["overall"] if m["significant"]}
    check("diff calls the rouge_l_f1 drop statistically significant", "reference.rouge_l_f1" in sig, f"significant: {sorted(sig) or 'none'}")

    print("== H5: cosmetic changes (UPPERCASE + trailing '.') should produce no flips")
    _, d2 = sagwa_json("diff", "--baseline", good, "--candidate", cosmetic, "--gates-config", str(GATES))
    cosmetic_flips = sorted(f["case_id"] for f in d2["flips"])
    check("cosmetic run has zero flips", not cosmetic_flips, f"{len(cosmetic_flips)} flipped: {cosmetic_flips}" if cosmetic_flips else "")

    print("== H6: do failure clusters line up with fault types? (informational)")
    _, clusters = sagwa_json("cluster", "--run-id", bad, "--gates-config", str(GATES), "--min-cluster-size", "2")
    for c in clusters:
        faults = sorted({INJECTED_FAULTS.get(cid, ("<not injected>",))[0] for cid in c["case_ids"]})
        print(f"  cluster {c['cluster_id']:>2}: {c['case_ids']} -> fault types {faults}")

    print("\n" + "=" * 70)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print("=" * 70)
    print(f"{sum(ok for _, ok, _ in checks)}/{len(checks)} hypotheses held")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
