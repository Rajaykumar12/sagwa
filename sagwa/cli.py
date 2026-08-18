"""`sagwa run|diff|gate` — see PRD.md §5.3, §5.6, §5.8."""
import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import typer

from sagwa.adapters.base import TargetAdapter
from sagwa.adapters.stub import StubAdapter
from sagwa.datasets import DatasetError, load_golden_set
from sagwa.metrics import compute_metrics
from sagwa.runner import run_cases
from sagwa.storage import Result, Run, get_session

app = typer.Typer()

# The only built-in, zero-config target. Any other target pipeline is
# plugged in via a dynamic `module.path:ClassName` import (see
# `_load_adapter_class`) — Sagwa's core never needs to know about it, so
# integrating a new ML project never requires editing this file.
BUILTIN_ADAPTERS: dict[str, Callable[[], TargetAdapter]] = {
    "stub": StubAdapter,
}


def _load_adapter_class(target: str) -> Callable[[], TargetAdapter]:
    """Resolves `target` to an adapter factory: a built-in name, or a
    `module.path:ClassName` string pointing at any class implementing
    `TargetAdapter` (PRD FR-3a — "zero required changes to the target's own
    codebase" cuts both ways: zero required changes to Sagwa's, either).
    See examples/adapters/README.md for a worked example (ringo)."""
    if target in BUILTIN_ADAPTERS:
        return BUILTIN_ADAPTERS[target]
    if ":" not in target:
        raise ValueError(
            f"Unknown target '{target}'. Built-ins: {', '.join(BUILTIN_ADAPTERS)}. "
            "To use your own adapter, pass --target 'module.path:ClassName' "
            "(see examples/adapters/README.md)."
        )
    # So a module path relative to the current project's own root resolves,
    # the same way `python -m` would, when invoked via the installed `sagwa`
    # console script (whose own location is not the caller's project root).
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_path, class_name = target.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _git_sha(repo_path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _sagwa_git_sha() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return _git_sha(repo_root) or "unknown"


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.command()
def run(
    target: str = typer.Option(
        ..., help="Built-in name (e.g. 'stub') or 'module.path:ClassName' for your own adapter"
    ),
    dataset: Path = typer.Option(..., help="Path to a golden-set JSONL file"),
    concurrency: int = typer.Option(5, help="Max concurrent adapter calls (PRD FR-5)"),
):
    """Run a golden set against a target pipeline (PRD FR-4)."""
    try:
        cases = load_golden_set(dataset)
    except DatasetError as e:
        typer.echo(str(e))
        raise typer.Exit(1)

    try:
        adapter = _load_adapter_class(target)()
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    except (ImportError, AttributeError) as e:
        typer.echo(f"Could not load adapter '{target}': {e}")
        raise typer.Exit(1)
    except KeyError as e:
        # A custom adapter's __init__ reading a required env var directly
        # (as the ringo example does) raises a bare KeyError if it's unset —
        # translate that into an actionable message rather than a stack trace.
        typer.echo(f"Missing required environment variable for target '{target}': {e}")
        raise typer.Exit(1)

    outcomes = run_cases(adapter, cases, max_concurrency=concurrency)
    failed = [o for o in outcomes if o.error is not None]

    with get_session() as session:
        run_row = Run(
            sagwa_git_sha=_sagwa_git_sha(),
            target_pipeline_git_sha=_target_pipeline_git_sha(adapter),
            target_name=target,
            model=_run_model_label(adapter),
            dataset_path=str(dataset),
            dataset_sha256=_dataset_sha256(dataset),
            status="running",
        )
        session.add(run_row)
        session.flush()  # populate run_row.id

        for outcome in outcomes:
            if outcome.error is not None:
                session.add(
                    Result(
                        run_id=run_row.id,
                        case_id=outcome.case.id,
                        input=outcome.case.input,
                        output="",
                        latency_ms=0,
                        error=outcome.error,
                    )
                )
                continue

            result = outcome.result
            session.add(
                Result(
                    run_id=run_row.id,
                    case_id=outcome.case.id,
                    input=outcome.case.input,
                    output=result.answer,
                    context=result.context,
                    latency_ms=result.latency_ms,
                    tokens=result.tokens,
                    cost_usd=result.cost_usd,
                    metrics_json=compute_metrics(outcome.case, result.answer, result.context),
                )
            )
        run_row.status = "completed"
        run_id = run_row.id

    typer.echo(
        f"Run {run_id}: {len(cases)} case(s) against target '{target}'"
        + (f" ({len(failed)} failed)" if failed else "")
    )


def _target_pipeline_git_sha(adapter: TargetAdapter) -> str | None:
    """Pin the target pipeline's own git SHA when it's an external,
    independently-versioned repo (PRD FR-7a). Duck-types on an optional
    `repo_path` attribute — any adapter for a git-versioned target can
    expose one; adapters that don't (e.g. `stub`) simply get `None` here."""
    repo_path = getattr(adapter, "repo_path", None)
    if repo_path is None:
        return None
    return _git_sha(repo_path)


def _run_model_label(adapter: TargetAdapter) -> str:
    """Duck-types on an optional `model_label` attribute, the same
    convention `_target_pipeline_git_sha` uses for `repo_path` — an adapter
    for a target with no single static model id (e.g. one that routes
    per-query across model tiers) can set this in its own `__init__`
    instead of Sagwa's core needing to special-case it by name."""
    return getattr(adapter, "model_label", "n/a")


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
