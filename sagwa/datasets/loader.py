"""JSONL loader/validator for golden sets (PRD FR-2)."""
import json
from pathlib import Path

from pydantic import ValidationError

from sagwa.datasets.schema import GoldenCase


class DatasetError(Exception):
    """Raised when a golden-set file fails validation.

    Carries every line's error at once (not just the first) so a user can
    fix a malformed file in one pass instead of one failure at a time.
    """


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path}: file not found")

    cases: list[GoldenCase] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e})")
            continue
        try:
            case = GoldenCase.model_validate(data)
        except ValidationError as e:
            errors.append(f"line {lineno}: {e}")
            continue
        if case.id in seen_ids:
            errors.append(f"line {lineno}: duplicate case id '{case.id}'")
            continue
        seen_ids.add(case.id)
        cases.append(case)

    if errors:
        raise DatasetError(f"{path}: {len(errors)} invalid row(s):\n" + "\n".join(errors))

    return cases
