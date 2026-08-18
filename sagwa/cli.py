"""`sagwa run|diff|gate` — see PRD.md §5.3, §5.6, §5.8."""
import hashlib
import subprocess
from pathlib import Path

import typer

from sagwa.adapters.base import TargetAdapter
from sagwa.adapters.stub import StubAdapter
from sagwa.datasets import DatasetError, load_golden_set
from sagwa.storage import Result, Run, get_session

app = typer.Typer()

ADAPTERS: dict[str, TargetAdapter] = {
    "stub": StubAdapter(),
}


def _sagwa_git_sha() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.command()
def run(
    target: str = typer.Option(..., help="Registered adapter name, e.g. 'stub'"),
    dataset: Path = typer.Option(..., help="Path to a golden-set JSONL file"),
):
    """Run a golden set against a target pipeline (PRD FR-4)."""
    if target not in ADAPTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(ADAPTERS)}")
        raise typer.Exit(1)

    try:
        cases = load_golden_set(dataset)
    except DatasetError as e:
        typer.echo(str(e))
        raise typer.Exit(1)

    adapter = ADAPTERS[target]

    with get_session() as session:
        run_row = Run(
            sagwa_git_sha=_sagwa_git_sha(),
            target_name=target,
            model="n/a",  # real model id once a non-stub adapter is wired up
            dataset_path=str(dataset),
            dataset_sha256=_dataset_sha256(dataset),
            status="running",
        )
        session.add(run_row)
        session.flush()  # populate run_row.id

        for case in cases:
            result = adapter.run(case.input)
            session.add(
                Result(
                    run_id=run_row.id,
                    case_id=case.id,
                    input=case.input,
                    output=result.answer,
                    context=result.context,
                    latency_ms=result.latency_ms,
                    tokens=result.tokens,
                    cost_usd=result.cost_usd,
                )
            )
        run_row.status = "completed"
        run_id = run_row.id

    typer.echo(f"Run {run_id}: {len(cases)} case(s) against target '{target}'")


@app.command()
def diff(
    baseline: str = typer.Option(..., help="Baseline run id"),
    candidate: str = typer.Option(..., help="Candidate run id"),
):
    """Diff two runs (PRD FR-16..FR-19). Not implemented yet — see PLAN.md §9 (Week 7-8)."""
    typer.echo("sagwa diff: not implemented yet — see PLAN.md §9 (Week 7-8)")
    raise typer.Exit(1)


@app.command()
def gate(
    run_id: str = typer.Option(..., help="Run id to gate"),
    config: Path = typer.Option(Path("config/gates.yaml"), help="Gate thresholds config"),
):
    """Gate a run against configured thresholds (PRD FR-23..FR-25). Not implemented yet — see PLAN.md §9 (Week 9-10)."""
    typer.echo("sagwa gate: not implemented yet — see PLAN.md §9 (Week 9-10)")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
