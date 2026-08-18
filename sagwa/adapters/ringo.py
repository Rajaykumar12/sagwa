"""Adapter for ringo (../ringo), the primary v1 target pipeline.

Deliberately in-process rather than over HTTP: ringo's `/chat/text` endpoint
pops `context` out of the response before returning it (see
`ringo/backend/main.py`, `text_chat()`), so faithfulness/context-precision
metrics would have nothing to score against over the wire. Calling
`pipeline.process_text` directly keeps the full result dict, including
context. See docs/adr/0003-ringo-adapter-in-process.md.

Not wired up yet — ringo has its own heavy dependency set (whisper,
chromadb, etc.) that needs to be importable on this process's path, and the
exact ringo -> AdapterResult field mapping needs to be checked against a
running ringo instance. Left as a skeleton so the contract (FR-3a, FR-7a) is
visible without half-integrating something untested.
"""
import os
import subprocess
import sys
from pathlib import Path

from sagwa.adapters.base import AdapterResult


def ringo_git_sha(repo_path: str | Path) -> str:
    """Pin ringo's own git SHA for the run record (PRD FR-7a)."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
    ).strip()


class RingoAdapter:
    name = "ringo"

    def __init__(self, repo_path: str | None = None):
        self.repo_path = Path(repo_path or os.environ["RINGO_REPO_PATH"])
        sys.path.insert(0, str(self.repo_path / "backend"))

    def run(self, case_input: str) -> AdapterResult:
        from pipeline import PipelineOrchestrator  # noqa: F401  (ringo's backend/pipeline.py)

        # TODO: instantiate ringo's pipeline and call process_text(case_input),
        # then map its result dict {response, context, sources, ...} onto
        # AdapterResult. Needs a live ringo instance (vectorstore, Groq key)
        # to validate the field mapping — deferred past this scaffolding pass.
        raise NotImplementedError(
            "RingoAdapter is a skeleton — see module docstring and "
            "docs/adr/0003-ringo-adapter-in-process.md"
        )
