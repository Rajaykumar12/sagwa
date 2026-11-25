"""Versioned prompt store (PRD FR-39).

A prompt is a directory of immutable versions plus a pointer to the one in use:

    <root>/<name>/v1.txt, v2.txt, ...   one file per version, never rewritten
    <root>/<name>/v2.json               where a version came from (parent, note)
    <root>/<name>/current               the version a target loads by default

`sagwa improve` only ever adds a version and, once a revision is accepted,
moves `current`. Every version that was ever evaluated stays on disk, so any
past run can be reproduced from `Run.prompt_version`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_VERSION = re.compile(r"^v(\d+)$")


class PromptStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def versions(self, name: str) -> list[str]:
        """Versions of `name`, oldest first."""
        found = [p.stem for p in (self.root / name).glob("v*.txt") if _VERSION.match(p.stem)]
        return sorted(found, key=lambda v: int(v[1:]))

    def current(self, name: str) -> str:
        pointer = self.root / name / "current"
        return pointer.read_text().strip() if pointer.exists() else self.versions(name)[-1]

    def load(self, name: str, version: str | None = None) -> str:
        version = version or self.current(name)
        path = self.root / name / f"{version}.txt"
        if not path.exists():
            raise KeyError(f"no prompt version {name}/{version} in {self.root}")
        return path.read_text()

    def add_version(self, name: str, text: str, parent: str | None = None, note: str = "") -> str:
        """Writes `text` as the next version and returns its id. Never touches an
        existing version, and doesn't move `current`."""
        existing = self.versions(name)
        version = f"v{int(existing[-1][1:]) + 1 if existing else 1}"
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{version}.txt"
        with path.open("x") as f:  # "x": fail rather than overwrite, even in a race
            f.write(text)
        (directory / f"{version}.json").write_text(json.dumps(
            {"parent": parent, "note": note, "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        return version

    def set_current(self, name: str, version: str) -> None:
        if version not in self.versions(name):
            raise KeyError(f"no prompt version {name}/{version} in {self.root}")
        (self.root / name / "current").write_text(version + "\n")


def format_versions(versions: dict[str, str]) -> str:
    """The `Run.prompt_version` string for a set of prompt versions, e.g.
    "planner=v1,writer=v2": stable order, so equal sets compare equal."""
    return ",".join(f"{name}={version}" for name, version in sorted(versions.items()))
